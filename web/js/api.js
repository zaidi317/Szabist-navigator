/**
 * api.js — thin fetch wrappers for the Vision Navigator Flask API.
 * All functions return parsed JSON or throw an Error with a human message.
 */

export const API_BASE = "";

async function _req(path, options = {}) {
  // ngrok free-tier bypass + signal we want JSON back
  const headers = new Headers(options.headers ?? {});
  headers.set("ngrok-skip-browser-warning", "1");
  headers.set("X-Requested-With", "XMLHttpRequest");

  let res;
  try {
    res = await fetch(API_BASE + path, { ...options, headers });
  } catch (err) {
    throw new Error("Cannot reach server (" + err.message + ")");
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.error ?? `HTTP ${res.status}`);
  return json;
}

// ── GET /api/health ──────────────────────────────────────────────────────────
export function getHealth() {
  return _req("/api/health");
}

// ── GET /api/locations ───────────────────────────────────────────────────────
// Returns { locations: [{node_id,label,x,y,floor,neighbors},...], count }
export function getLocations() {
  return _req("/api/locations");
}

// ── POST /api/identify ───────────────────────────────────────────────────────
// imageBlob: Blob (JPEG)
// Returns { node_id, label, confidence, coordinates:{x,y,floor}, ... }
export function identify(imageBlob) {
  const fd = new FormData();
  fd.append("image", imageBlob, "frame.jpg");
  return _req("/api/identify", { method: "POST", body: fd });
}

// ── POST /api/navigate ───────────────────────────────────────────────────────
// Returns { path, waypoints:[{node_id,label,x,y,floor}], total_distance_meters, ... }
export function navigate(fromNode, toNode) {
  return _req("/api/navigate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ from_node: fromNode, to_node: toNode }),
  });
}

// ── POST /api/full-navigate ──────────────────────────────────────────────────
// imageBlob: Blob, toNode: string
// Returns { location_identification, navigation, waypoints, current_coordinates }
export function fullNavigate(imageBlob, toNode) {
  const fd = new FormData();
  fd.append("image", imageBlob, "frame.jpg");
  fd.append("destination", toNode);
  return _req("/api/full-navigate", { method: "POST", body: fd });
}

// ── POST /api/calibrate ──────────────────────────────────────────────────────
// frames: [{blob: Blob, alpha: number}]  (N ≥ 3)
// Returns { node_id, label, anchor_alpha, confidence, heading_offset_deg, votes }
export function calibrate(frames) {
  const fd = new FormData();
  frames.forEach(({ blob, alpha }, i) => {
    fd.append("images[]", blob, `frame${i}.jpg`);
    fd.append("alphas[]", String(alpha));
  });
  return _req("/api/calibrate", { method: "POST", body: fd });
}

// ── POST /api/zabgpt ─────────────────────────────────────────────────────────
export function askZabGPT(query) {
  return _req("/api/zabgpt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
}
