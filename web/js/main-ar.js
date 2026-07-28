/**
 * main-ar.js — AR session orchestrator for Vision Navigator.
 *
 * Startup sequence (see init()):
 *  1. Read destination from ?dest= param or state.js
 *  2. Verify calibration exists; redirect to calibrate.html if not
 *  3. Restore Heading from saved anchor_alpha
 *  4. Load all campus nodes → initialise MiniMap
 *  5. Initialise AROverlay on the canvas
 *  6. Show tap-to-start prompt
 *
 * After user tap (startAR()):
 *  7. Start rear camera (getUserMedia — inside click handler)
 *  8. Start DeviceOrientation listener → heading.update() on every event
 *  9. Fire initial /api/navigate (we know current node from calibration)
 * 10. Begin re-localisation loop every 3500 ms:
 *       captureJpeg → POST /api/identify → if node changed, re-navigate
 * 11. requestAnimationFrame loop → redraws AR overlay every frame
 * 12. MiniMap redrawn after every re-localise tick
 */

import { startCamera, captureJpeg, stopCamera } from "./camera.js";
import { Sensors }    from "./sensors.js";
import { Heading }    from "./heading.js";
import { AROverlay }  from "./ar-overlay.js";
import { MiniMap }    from "./minimap.js";
import { getLocations, identify, navigate } from "./api.js";
import { state }      from "./state.js";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const videoEl       = document.getElementById("ar-video");
const arCanvas      = document.getElementById("ar-canvas");
const minimapCanvas = document.getElementById("minimap-canvas");
const tapPrompt     = document.getElementById("tap-prompt");
const arrivedOverlay = document.getElementById("arrived-overlay");
const arrivedLabel  = document.getElementById("arrived-label");
const destBadge     = document.getElementById("dest-badge");
const banner        = document.getElementById("banner");

// ── session state ────────────────────────────────────────────────────────────
const heading  = new Heading();
const sensors  = new Sensors();
let overlay    = null;   // AROverlay (created after DOM ready)
let minimap    = null;   // MiniMap
let allNodes   = [];     // [{node_id, label, x, y, floor, neighbors}]
let nodeMap    = {};     // node_id → node object

let destNodeId    = null;
let currentNodeId = null;
let waypoints     = [];  // ordered [{node_id,label,x,y,floor}] from API
let currentWpIdx  = 0;   // index of the node the user is currently AT in waypoints[]

const RELOCALISE_MS = 3500;
let _relocaliseTimer = null;
let _animFrameId    = null;
let _sessionActive  = false;

// ── helpers ───────────────────────────────────────────────────────────────────

function showBanner(msg, type = "error") {
  banner.textContent = msg;
  banner.className   = `banner show ${type}`;
}

function hideBanner() {
  banner.className = "banner";
}

/**
 * Bearing from current node to next node in campus-map space.
 * Convention: 0° = north (+Y), 90° = east (+X), same as compass bearing.
 * @param {{x:number, y:number}} from
 * @param {{x:number, y:number}} to
 * @returns {number} degrees 0–360
 */
function bearingBetween(from, to) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  // atan2(dx, dy): 0 when dy>0 (north), 90 when dx>0 (east)
  const deg = Math.atan2(dx, dy) * (180 / Math.PI);
  return (deg + 360) % 360;
}

/** Straight-line distance in metres between two nodes. */
function distanceBetween(a, b) {
  return Math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2);
}

/** Total remaining path distance (sum of edge distances from currentWpIdx). */
function remainingDistance() {
  let d = 0;
  for (let i = currentWpIdx; i < waypoints.length - 1; i++) {
    d += distanceBetween(waypoints[i], waypoints[i + 1]);
  }
  return d;
}

/** Advance currentWpIdx to the position of nodeId in waypoints (if found). */
function advanceToNode(nodeId) {
  const idx = waypoints.findIndex(w => w.node_id === nodeId);
  if (idx !== -1 && idx >= currentWpIdx) {
    currentWpIdx = idx;
  }
}

// ── overlay redraw (runs every animation frame) ───────────────────────────────

function drawFrame() {
  if (!_sessionActive) return;
  _animFrameId = requestAnimationFrame(drawFrame);

  if (!overlay) return;

  const arrived = currentNodeId === destNodeId;

  if (arrived) {
    overlay.draw({ turnAngle: 0, distanceM: 0, nextLabel: "", isArrived: true });
    showArrived();
    return;
  }

  // Next waypoint is the one AFTER where we currently are
  const nextWp = waypoints[currentWpIdx + 1] ?? waypoints[currentWpIdx];
  if (!nextWp) return;

  const currNode = nodeMap[currentNodeId] ?? waypoints[currentWpIdx];
  if (!currNode) return;

  const bearing   = bearingBetween(currNode, nextWp);
  const turnAngle = heading.turnAngleTo(bearing);
  const distM     = remainingDistance();

  overlay.draw({
    turnAngle,
    distanceM: distM,
    nextLabel: nextWp.label,
    isArrived: false,
  });
}

function showArrived() {
  arrivedOverlay.classList.add("show");
  const dest = state.getDestination();
  arrivedLabel.textContent = dest?.label ?? destNodeId ?? "your destination";
  stopRelocalise();
}

// ── re-localisation loop (every 3500 ms) ─────────────────────────────────────

async function relocalise() {
  if (!_sessionActive) return;

  let blob;
  try {
    blob = await captureJpeg(videoEl, 0.7);
  } catch {
    return; // camera not ready yet — skip this tick
  }
  if (!blob) return;

  let result;
  try {
    result = await identify(blob);
  } catch (err) {
    showBanner("Cannot reach server — check WiFi connection.", "error");
    return;
  }

  hideBanner();

  const identified = result.node_id ?? result.location;
  if (!identified) return;

  // Node changed → re-navigate and update minimap
  if (identified !== currentNodeId) {
    currentNodeId = identified;
    state.setCurrentNode(identified);

    // Advance pointer into the existing waypoints array if possible
    advanceToNode(identified);

    // If identified node is not in path at all → full re-navigate
    const inPath = waypoints.some(w => w.node_id === identified);
    if (!inPath && destNodeId) {
      try {
        const navResult = await navigate(identified, destNodeId);
        waypoints    = navResult.waypoints ?? [];
        currentWpIdx = 0;
        state.setPath(waypoints);
      } catch {
        // keep old waypoints
      }
    }
  }

  // Redraw minimap on every localise tick (even if node didn't change)
  if (minimap) {
    minimap.update({
      currentNodeId,
      destinationNodeId: destNodeId,
      path: waypoints.map(w => w.node_id),
    });
  }
}

function startRelocalise() {
  relocalise(); // fire immediately
  _relocaliseTimer = setInterval(relocalise, RELOCALISE_MS);
}

function stopRelocalise() {
  clearInterval(_relocaliseTimer);
  _relocaliseTimer = null;
}

// ── initial navigation ────────────────────────────────────────────────────────

async function fetchInitialRoute() {
  if (!currentNodeId || !destNodeId) return;

  if (currentNodeId === destNodeId) {
    showArrived();
    return;
  }

  try {
    const result = await navigate(currentNodeId, destNodeId);
    waypoints    = result.waypoints ?? [];
    currentWpIdx = 0;
    state.setPath(waypoints);

    if (minimap) {
      minimap.update({
        currentNodeId,
        destinationNodeId: destNodeId,
        path: waypoints.map(w => w.node_id),
      });
    }
  } catch (err) {
    showBanner(`Route unavailable: ${err.message}`);
  }
}

// ── startAR (called from tap handler — inside user gesture) ──────────────────

async function startAR() {
  tapPrompt.style.display = "none";

  // 1. Sensor permission (iOS gate)
  const sensorOk = await sensors.requestPermission();
  if (!sensorOk) {
    showBanner("Motion sensor access denied. Heading will not update.");
  } else {
    sensors.start(({ alpha }) => {
      heading.update(alpha);
    });
  }

  // 2. Camera
  try {
    await startCamera(videoEl);
  } catch {
    showBanner("Camera access denied. Please allow camera permission.");
    tapPrompt.style.display = "";
    return;
  }

  _sessionActive = true;

  // 3. Initial route
  await fetchInitialRoute();

  // 4. Animation frame loop (draws overlay every frame)
  drawFrame();

  // 5. Re-localisation loop
  startRelocalise();
}

// ── init ──────────────────────────────────────────────────────────────────────

async function init() {
  // Read destination from ?dest= URL param or state.js
  const params = new URLSearchParams(window.location.search);
  destNodeId   = params.get("dest") ?? state.getDestination()?.node_id ?? null;

  if (!destNodeId) {
    showBanner("No destination selected. Please go back and pick one.");
    return;
  }

  // Check calibration
  const cal = state.getCalibration();
  if (!cal || typeof cal.anchor_alpha !== "number") {
    // Redirect to calibrate, preserving destination
    window.location.href = `calibrate.html?dest=${encodeURIComponent(destNodeId)}`;
    return;
  }

  // Restore current node from state (set by calibration)
  currentNodeId = state.getCurrentNode();

  // Apply saved heading anchor
  heading.calibrate(cal.anchor_alpha);

  // Update destination badge
  const dest = state.getDestination();
  if (destBadge) destBadge.textContent = dest?.label ?? destNodeId;

  // Load all campus nodes for MiniMap
  try {
    const data = await getLocations();
    allNodes = data.locations ?? [];
    nodeMap  = Object.fromEntries(allNodes.map(n => [n.node_id, n]));
  } catch {
    showBanner("Cannot load campus map — is Flask running?", "error");
    // Non-fatal: AR still works without minimap
  }

  // Initialise canvas renderers
  overlay = new AROverlay(arCanvas);
  if (allNodes.length) {
    minimap = new MiniMap(minimapCanvas, allNodes);
    // Draw initial minimap with calibrated position
    minimap.update({
      currentNodeId,
      destinationNodeId: destNodeId,
      path: state.getPath().map(w => w.node_id),
    });
  }

  // Wire tap-to-start prompt
  tapPrompt.addEventListener("click", () => {
    startAR().catch(err => showBanner(err.message));
  }, { once: true });
}

// ── cleanup on page unload ────────────────────────────────────────────────────

window.addEventListener("pagehide", () => {
  _sessionActive = false;
  cancelAnimationFrame(_animFrameId);
  stopRelocalise();
  sensors.stop();
  stopCamera();
  minimap?.destroy();
});

// ── boot ──────────────────────────────────────────────────────────────────────
init().catch(err => showBanner(err.message));
