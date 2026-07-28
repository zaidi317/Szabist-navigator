/**
 * state.js — sessionStorage-backed app state
 *
 * Keys stored:
 *   vn_destination   {node_id, label}
 *   vn_path          [{node_id, label, x, y, floor}, ...]
 *   vn_calibration   {node_id, anchor_alpha, heading_offset_deg}
 *   vn_current_node  string
 */

const PREFIX = "vn_";

function _key(k) { return PREFIX + k; }

export const state = {
  // ── setters ──────────────────────────────────────────────────────────────
  setDestination(nodeId, label) {
    sessionStorage.setItem(_key("destination"), JSON.stringify({ node_id: nodeId, label }));
  },

  setPath(waypoints) {
    sessionStorage.setItem(_key("path"), JSON.stringify(waypoints));
  },

  setCalibration(nodeId, anchorAlpha, headingOffsetDeg = 0) {
    sessionStorage.setItem(_key("calibration"), JSON.stringify({
      node_id: nodeId,
      anchor_alpha: anchorAlpha,
      heading_offset_deg: headingOffsetDeg,
    }));
  },

  setCurrentNode(nodeId) {
    sessionStorage.setItem(_key("current_node"), nodeId);
  },

  // ── getters ──────────────────────────────────────────────────────────────
  getDestination() {
    try { return JSON.parse(sessionStorage.getItem(_key("destination"))); } catch { return null; }
  },

  getPath() {
    try { return JSON.parse(sessionStorage.getItem(_key("path"))) ?? []; } catch { return []; }
  },

  getCalibration() {
    try { return JSON.parse(sessionStorage.getItem(_key("calibration"))); } catch { return null; }
  },

  getCurrentNode() {
    return sessionStorage.getItem(_key("current_node")) ?? null;
  },

  // ── helpers ───────────────────────────────────────────────────────────────
  isCalibrated() {
    const c = this.getCalibration();
    return c !== null && typeof c.anchor_alpha === "number";
  },

  clearAll() {
    ["destination", "path", "calibration", "current_node"].forEach(k =>
      sessionStorage.removeItem(_key(k))
    );
  },
};
