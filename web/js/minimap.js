/**
 * minimap.js — always-visible Canvas mini-map renderer.
 *
 * Draws a 2D top-down view of the campus graph showing:
 *   • All nodes as labelled circles (grey)
 *   • The current path edges as dashed coral lines
 *   • The current position node (blue fill)
 *   • The destination node (red fill)
 *   • A blinking position dot on the current node
 *
 * All (x, y) campus-metre coordinates are normalised to fit the canvas
 * with padding on every update, so it works regardless of coordinate scale.
 */

const PAD        = 18;   // canvas padding in px
const NODE_R     = 7;    // node circle radius px
const FONT       = "bold 9px -apple-system, sans-serif";

// Colours
const C_EDGE     = "#D1D5DB";  // grey — non-path edges
const C_PATH     = "#FF6B4A";  // coral — path edges
const C_NODE     = "#9CA3AF";  // grey — default node fill
const C_CURRENT  = "#3B82F6";  // blue  — current node
const C_DEST     = "#EF4444";  // red   — destination
const C_LABEL    = "#374151";  // dark  — node labels
const C_BG       = "rgba(255,255,255,0.92)";

export class MiniMap {
  /**
   * @param {HTMLCanvasElement} canvasEl  — #minimap-canvas
   * @param {Array<object>}     allNodes  — from GET /api/locations
   *   Each: { node_id, label, x, y, floor, neighbors }
   */
  constructor(canvasEl, allNodes) {
    this._canvas   = canvasEl;
    this._ctx      = canvasEl.getContext("2d");
    this._allNodes = allNodes;          // full node list (static)
    this._nodeMap  = Object.fromEntries(allNodes.map(n => [n.node_id, n]));

    // Pre-compute scale so the whole graph fits inside the canvas
    this._scale    = this._computeScale(allNodes, canvasEl.width, canvasEl.height);

    this._blinkOn  = true;
    this._blinkTimer = setInterval(() => {
      this._blinkOn = !this._blinkOn;
    }, 600);
  }

  /** Free the blink interval when the map is no longer needed. */
  destroy() {
    clearInterval(this._blinkTimer);
  }

  /**
   * Redraw the mini-map.
   * @param {object} opts
   * @param {string}   opts.currentNodeId      node the user is at
   * @param {string}   opts.destinationNodeId  target node
   * @param {string[]} opts.path               ordered list of node_ids
   */
  update({ currentNodeId, destinationNodeId, path = [] }) {
    const { _ctx: ctx, _canvas: cv } = this;
    const pathSet = new Set(path); // fast lookup

    // Build set of path edges (both directions) for highlighted drawing
    const pathEdges = new Set();
    for (let i = 0; i < path.length - 1; i++) {
      pathEdges.add(`${path[i]}|${path[i + 1]}`);
      pathEdges.add(`${path[i + 1]}|${path[i]}`);
    }

    // ── background ─────────────────────────────────────────────────────────
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = C_BG;
    _roundRect(ctx, 0, 0, cv.width, cv.height, 8);
    ctx.fill();

    // ── edges ───────────────────────────────────────────────────────────────
    ctx.lineWidth = 2;
    for (const node of this._allNodes) {
      const p1 = this._toCanvas(node);
      for (const neighborId of node.neighbors) {
        const neighbor = this._nodeMap[neighborId];
        if (!neighbor) continue;
        // Draw each undirected edge once (only when node_id < neighbor)
        if (node.node_id >= neighborId) continue;
        const p2 = this._toCanvas(neighbor);

        const isPathEdge = pathEdges.has(`${node.node_id}|${neighborId}`);
        if (isPathEdge) {
          ctx.setLineDash([5, 3]);
          ctx.strokeStyle = C_PATH;
          ctx.lineWidth   = 2.5;
        } else {
          ctx.setLineDash([]);
          ctx.strokeStyle = C_EDGE;
          ctx.lineWidth   = 1.5;
        }

        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    // ── nodes ────────────────────────────────────────────────────────────────
    for (const node of this._allNodes) {
      const { x, y } = this._toCanvas(node);
      const isCurrent = node.node_id === currentNodeId;
      const isDest    = node.node_id === destinationNodeId;

      let fill = C_NODE;
      if (isCurrent && this._blinkOn) fill = C_CURRENT;
      if (isDest)                      fill = C_DEST;
      // Destination overrides current if same node (arrived)
      if (isCurrent && isDest)         fill = "#22C55E"; // green

      // Circle
      ctx.beginPath();
      ctx.arc(x, y, NODE_R, 0, Math.PI * 2);
      ctx.fillStyle   = fill;
      ctx.shadowColor = isCurrent || isDest ? "rgba(0,0,0,0.25)" : "transparent";
      ctx.shadowBlur  = isCurrent || isDest ? 6 : 0;
      ctx.fill();
      ctx.shadowBlur = 0;

      // White border ring for emphasis on special nodes
      if (isCurrent || isDest) {
        ctx.beginPath();
        ctx.arc(x, y, NODE_R, 0, Math.PI * 2);
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth   = 2;
        ctx.stroke();
      }

      // Label — short version (first 6 chars)
      const shortLabel = node.label.split(" ")[0].slice(0, 7);
      ctx.font      = FONT;
      ctx.fillStyle = C_LABEL;
      ctx.textAlign = "center";
      ctx.fillText(shortLabel, x, y + NODE_R + 9);
    }

    // ── floor indicator (if multi-floor path) ────────────────────────────────
    const floors = [...new Set(path.map(id => this._nodeMap[id]?.floor ?? 0))];
    if (floors.length > 1) {
      ctx.font      = "10px -apple-system, sans-serif";
      ctx.fillStyle = "#6B7280";
      ctx.textAlign = "right";
      ctx.fillText("^ multi-floor", cv.width - 6, cv.height - 5);
    }
  }

  // ── private ────────────────────────────────────────────────────────────────

  /** Map campus (x, y) metres to canvas pixels, respecting padding. */
  _toCanvas(node) {
    const { scale, minX, minY } = this._scale;
    return {
      x: PAD + (node.x - minX) * scale,
      y: PAD + (node.y - minY) * scale,
    };
  }

  _computeScale(nodes, canvasW, canvasH) {
    if (!nodes.length) return { scale: 1, minX: 0, minY: 0 };

    const xs = nodes.map(n => n.x);
    const ys = nodes.map(n => n.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const scale = Math.min(
      (canvasW - PAD * 2) / rangeX,
      (canvasH - PAD * 2 - 12) / rangeY  // extra bottom room for labels
    );

    return { scale, minX, minY };
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
