/**
 * ar-overlay.js — Canvas AR arrow renderer.
 *
 * Draws on top of the live <video> feed:
 *   • A directional arrow that rotates with the user's heading
 *   • A distance label (metres) at the bottom
 *   • The next waypoint label at the top
 *
 * Coordinate convention:
 *   turnAngle = 0   → arrow points straight up ("keep going forward")
 *   turnAngle > 0   → turn right
 *   turnAngle < 0   → turn left
 *   Range: −180 to +180 degrees
 *
 * The canvas is sized to match its CSS container on every draw call (no fixed dims).
 */

// EMA to smooth the visual arrow rotation independently of heading.js
const VISUAL_EMA = 0.18;

export class AROverlay {
  /**
   * @param {HTMLCanvasElement} canvasEl  — #ar-canvas
   */
  constructor(canvasEl) {
    this._canvas  = canvasEl;
    this._ctx     = canvasEl.getContext("2d");
    this._smoothedAngle = 0; // degrees, EMA-smoothed for rendering
  }

  /** Sync canvas pixel size to its CSS layout size. Call before draw(). */
  resize() {
    const { offsetWidth: w, offsetHeight: h } = this._canvas;
    if (this._canvas.width !== w || this._canvas.height !== h) {
      this._canvas.width  = w;
      this._canvas.height = h;
    }
  }

  /**
   * Render one frame of the AR overlay.
   * @param {object} opts
   * @param {number}  opts.turnAngle   Degrees to turn (−180..+180); 0 = straight ahead
   * @param {number}  opts.distanceM   Distance to next waypoint in metres
   * @param {string}  opts.nextLabel   Display name of the next node
   * @param {boolean} [opts.isArrived] If true, draw an arrival indicator instead
   */
  draw({ turnAngle = 0, distanceM = 0, nextLabel = "", isArrived = false }) {
    this.resize();
    const { _ctx: ctx, _canvas: cv } = this;
    const cx = cv.width  / 2;
    const cy = cv.height / 2;

    ctx.clearRect(0, 0, cv.width, cv.height);

    if (isArrived) {
      this._drawArrived(ctx, cx, cy);
      return;
    }

    // Smooth visual angle with EMA (wrap-aware)
    let delta = turnAngle - this._smoothedAngle;
    if (delta >  180) delta -= 360;
    if (delta < -180) delta += 360;
    this._smoothedAngle = this._smoothedAngle + VISUAL_EMA * delta;

    const angleRad = (this._smoothedAngle * Math.PI) / 180;

    // ── arrow ──────────────────────────────────────────────────────────────
    ctx.save();
    ctx.translate(cx, cy - 30); // shift arrow toward upper-centre
    ctx.rotate(angleRad);
    this._drawArrow(ctx);
    ctx.restore();

    // ── distance label ─────────────────────────────────────────────────────
    const distText = distanceM >= 1000
      ? `${(distanceM / 1000).toFixed(1)} km`
      : `${Math.round(distanceM)} m`;

    ctx.save();
    ctx.font         = "bold 22px -apple-system, sans-serif";
    ctx.fillStyle    = "#FFFFFF";
    ctx.textAlign    = "center";
    ctx.shadowColor  = "rgba(0,0,0,0.6)";
    ctx.shadowBlur   = 6;
    ctx.fillText(distText, cx, cv.height - 40);
    ctx.restore();

    // ── next waypoint label ────────────────────────────────────────────────
    if (nextLabel) {
      ctx.save();
      ctx.font        = "600 16px -apple-system, sans-serif";
      ctx.fillStyle   = "#FFFFFF";
      ctx.textAlign   = "center";
      ctx.shadowColor = "rgba(0,0,0,0.7)";
      ctx.shadowBlur  = 8;
      // pill background
      const labelW = ctx.measureText(nextLabel).width + 28;
      ctx.fillStyle = "rgba(30,42,74,0.78)"; // --color-navy
      _roundRect(ctx, cx - labelW / 2, 20, labelW, 34, 8);
      ctx.fill();
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(nextLabel, cx, 43);
      ctx.restore();
    }

    // ── turn hint text (only when turn > 20°) ─────────────────────────────
    const absTurn = Math.abs(this._smoothedAngle);
    if (absTurn > 20) {
      const hint = this._smoothedAngle > 0 ? "Turn right →" : "← Turn left";
      ctx.save();
      ctx.font      = "500 14px -apple-system, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.textAlign = "center";
      ctx.shadowColor = "rgba(0,0,0,0.5)";
      ctx.shadowBlur  = 4;
      ctx.fillText(hint, cx, cy + 80);
      ctx.restore();
    }
  }

  clear() {
    const { _ctx: ctx, _canvas: cv } = this;
    ctx.clearRect(0, 0, cv.width, cv.height);
  }

  // ── private ───────────────────────────────────────────────────────────────

  _drawArrow(ctx) {
    const size = 80;
    // Outer coral circle
    ctx.beginPath();
    ctx.arc(0, 0, size * 0.72, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,107,74,0.88)"; // --color-coral
    ctx.shadowColor = "rgba(0,0,0,0.4)";
    ctx.shadowBlur  = 16;
    ctx.fill();
    ctx.shadowBlur  = 0;

    // White upward arrow (chevron-style)
    ctx.beginPath();
    ctx.moveTo(0,            -size * 0.45); // tip
    ctx.lineTo( size * 0.35,  size * 0.28); // bottom-right
    ctx.lineTo( size * 0.15,  size * 0.12); // inner-right
    ctx.lineTo(-size * 0.15,  size * 0.12); // inner-left
    ctx.lineTo(-size * 0.35,  size * 0.28); // bottom-left
    ctx.closePath();
    ctx.fillStyle = "#FFFFFF";
    ctx.fill();
  }

  _drawArrived(ctx, cx, cy) {
    // Green check circle — the green overlay in CSS handles the full-screen
    // tint; here we just draw a centred check mark for the canvas layer.
    ctx.save();
    ctx.font      = "80px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("✅", cx, cy + 28);
    ctx.restore();
  }
}

// ── utility ───────────────────────────────────────────────────────────────────

function _roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
