/**
 * heading.js — gyroscope-based relative heading state machine.
 *
 * Indoor compass bearings are unreliable, so we use relative heading:
 *
 *   relativeHeading = ((currentAlpha - anchorAlpha) + 360) % 360
 *
 * where anchorAlpha is the device's alpha reading at calibration time.
 * The value represents "how many degrees the user has turned since calibration."
 *
 * An exponential moving average (EMA, α=0.1) smooths jitter from the gyro.
 *
 * States:
 *   UNCALIBRATED — anchor not set, heading is always 0
 *   CALIBRATED   — producing live relative headings
 */

const EMA_ALPHA = 0.1; // smoothing factor (lower = smoother but slower)

export class Heading {
  constructor() {
    this._state       = "UNCALIBRATED";
    this._anchorAlpha = 0;
    this._smoothed    = 0; // EMA-smoothed relative heading (degrees)
    this._raw         = 0; // last raw relative heading
  }

  get state()   { return this._state; }
  get isReady() { return this._state === "CALIBRATED"; }

  /**
   * Set the calibration anchor. Switches state to CALIBRATED.
   * @param {number} anchorAlpha  device alpha at calibration moment (degrees)
   */
  calibrate(anchorAlpha) {
    this._anchorAlpha = anchorAlpha;
    this._smoothed    = 0;
    this._raw         = 0;
    this._state       = "CALIBRATED";
  }

  /** Reset to uncalibrated (called when a new AR session starts). */
  reset() {
    this._state = "UNCALIBRATED";
    this._smoothed = 0;
    this._raw = 0;
  }

  /**
   * Feed a new alpha reading from the DeviceOrientation event.
   * Updates the EMA-smoothed heading. Call this on every orientation event.
   * @param {number} alpha  current device alpha (degrees 0–360)
   */
  update(alpha) {
    // Raw relative heading (always positive)
    const raw = ((alpha - this._anchorAlpha) + 360) % 360;
    this._raw = raw;

    if (this._state !== "CALIBRATED") return;

    // EMA — handle wrap-around at 0°/360° boundary
    let delta = raw - this._smoothed;
    if (delta >  180) delta -= 360;
    if (delta < -180) delta += 360;

    this._smoothed = (this._smoothed + EMA_ALPHA * delta + 360) % 360;
  }

  /**
   * Get the current smoothed relative heading in degrees (0–360).
   * Returns 0 if not calibrated.
   */
  getHeading() {
    return this._state === "CALIBRATED" ? this._smoothed : 0;
  }

  /** Raw (un-smoothed) relative heading — useful for debugging. */
  getRawHeading() {
    return this._raw;
  }

  /**
   * Given a bearing TO the next waypoint (degrees, 0 = forward at calibration),
   * compute how many degrees the user must turn (−180 to +180, negative = left).
   * @param {number} waypointBearing  direction to next waypoint in relative coords
   * @returns {number}
   */
  turnAngleTo(waypointBearing) {
    let diff = waypointBearing - this.getHeading();
    if (diff >  180) diff -= 360;
    if (diff < -180) diff += 360;
    return diff;
  }
}
