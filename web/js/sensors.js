/**
 * sensors.js — DeviceOrientationEvent wrapper with iOS 13+ permission flow.
 *
 * Usage:
 *   import { Sensors } from "./sensors.js";
 *   const sensors = new Sensors();
 *   await sensors.requestPermission();   // call from a click handler on iOS
 *   sensors.start(({ alpha, beta, gamma }) => { ... });
 *   sensors.stop();
 *
 * alpha: compass bearing 0–360 (yaw around Z axis).
 *        Note: this is device-relative, NOT north-aligned — heading.js
 *        converts it to a relative heading using the calibration anchor.
 * beta:  front-to-back tilt –180–180 (pitch around X axis).
 * gamma: left-right tilt  –90–90  (roll  around Y axis).
 */

export class Sensors {
  constructor() {
    this._handler = null;
    this._callback = null;
    this._available = "DeviceOrientationEvent" in window;
  }

  get available() { return this._available; }

  /**
   * Request iOS 13+ permission if needed.
   * Must be called synchronously inside a user gesture (button click).
   * Returns true if permission is granted or not required (Android/desktop).
   * @returns {Promise<boolean>}
   */
  async requestPermission() {
    if (
      typeof DeviceOrientationEvent !== "undefined" &&
      typeof DeviceOrientationEvent.requestPermission === "function"
    ) {
      try {
        const result = await DeviceOrientationEvent.requestPermission();
        return result === "granted";
      } catch {
        return false;
      }
    }
    // Android / desktop — permission not required
    return true;
  }

  /**
   * Start listening for orientation events.
   * @param {(data: {alpha:number, beta:number, gamma:number}) => void} callback
   */
  start(callback) {
    this._callback = callback;

    this._handler = (evt) => {
      // alpha can be null on some browsers/devices; default to 0
      callback({
        alpha: evt.alpha ?? 0,
        beta:  evt.beta  ?? 0,
        gamma: evt.gamma ?? 0,
      });
    };

    window.addEventListener("deviceorientation", this._handler, true);
  }

  /** Stop listening. */
  stop() {
    if (this._handler) {
      window.removeEventListener("deviceorientation", this._handler, true);
      this._handler = null;
    }
    this._callback = null;
  }
}
