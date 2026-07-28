/**
 * calibrate.js — calibration pan flow for Vision Navigator AR.
 *
 * Flow:
 *  1. User taps "Start Calibration"  (click handler — required for iOS camera + sensor gates)
 *  2. Request DeviceOrientation permission (iOS 13+)
 *  3. Start rear camera  → show preview
 *  4. Capture 5 JPEG frames at 700 ms intervals while recording device alpha each time
 *  5. Animate the progress ring as frames are captured
 *  6. POST frames + alphas to /api/calibrate
 *  7. Persist { node_id, anchor_alpha, heading_offset_deg } in sessionStorage via state.js
 *  8. Redirect to ar.html?dest=<node_id>
 */

import { startCamera, stopCamera, captureJpeg } from "./camera.js";
import { Sensors }  from "./sensors.js";
import { calibrate } from "./api.js";
import { state }    from "./state.js";

// ── constants ────────────────────────────────────────────────────────────────
const FRAME_COUNT    = 5;
const FRAME_INTERVAL = 700; // ms between captures
const RING_CIRC      = 326.73; // 2π × r52 — matches SVG stroke-dasharray

// ── DOM refs ─────────────────────────────────────────────────────────────────
const videoEl   = document.getElementById("cal-video");
const ringFill  = document.getElementById("ring-fill");
const ringLabel = document.getElementById("ring-label");
const calTitle  = document.getElementById("cal-title");
const calHint   = document.getElementById("cal-hint");
const btnCal    = document.getElementById("btn-cal");
const banner    = document.getElementById("banner");

// ── helpers ──────────────────────────────────────────────────────────────────
function showBanner(msg, type = "error") {
  banner.textContent = msg;
  banner.className   = `banner show ${type}`;
}

function hideBanner() {
  banner.className = "banner";
}

function setRing(framesCapured) {
  const pct    = framesCapured / FRAME_COUNT;
  const offset = RING_CIRC * (1 - pct);
  ringFill.style.strokeDashoffset = offset;
  ringLabel.textContent = Math.round(pct * 100) + "%";
}

function setStatus(title, hint) {
  calTitle.textContent = title;
  calHint.textContent  = hint;
}

// Read destination from URL param first, fallback to sessionStorage
function getDestNodeId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("dest") ?? state.getDestination()?.node_id ?? null;
}

// ── main ─────────────────────────────────────────────────────────────────────
const sensors = new Sensors();

// Latest alpha value from the gyro; updated continuously once sensors start
let _currentAlpha = 0;
let _sensorsStarted = false;

btnCal.addEventListener("click", async () => {
  btnCal.disabled = true;
  hideBanner();

  // ── step 1: sensor permission (iOS gate — must be inside click handler) ──
  if (!_sensorsStarted) {
    const granted = await sensors.requestPermission();
    if (!granted) {
      showBanner("Sensor permission denied. Please allow motion access and try again.");
      btnCal.disabled = false;
      return;
    }
    sensors.start(({ alpha }) => { _currentAlpha = alpha; });
    _sensorsStarted = true;
  }

  // ── step 2: camera (must also be inside click handler) ───────────────────
  setStatus("Starting camera…", "Please wait.");
  try {
    await startCamera(videoEl);
    videoEl.style.display = "block";
  } catch (err) {
    showBanner("Camera access denied. Please allow camera permission and try again.");
    btnCal.disabled = false;
    return;
  }

  // Short pause so the camera auto-exposure stabilises
  await new Promise(r => setTimeout(r, 500));

  // ── step 3: capture frames ────────────────────────────────────────────────
  setStatus("Slowly pan around…", "Keep the camera moving while we scan your surroundings.");
  setRing(0);

  const frames = []; // [{blob, alpha}]

  for (let i = 0; i < FRAME_COUNT; i++) {
    await new Promise(r => setTimeout(r, FRAME_INTERVAL));

    const blob = await captureJpeg(videoEl, 0.7);
    if (!blob) {
      showBanner("Failed to capture frame. Please try again.");
      stopCamera();
      videoEl.style.display = "none";
      btnCal.disabled = false;
      return;
    }

    frames.push({ blob, alpha: _currentAlpha });
    setRing(i + 1);
  }

  // ── step 4: POST to /api/calibrate ───────────────────────────────────────
  setStatus("Identifying location…", "Checking your surroundings.");
  let calResult;
  try {
    calResult = await calibrate(frames);
  } catch (err) {
    showBanner(`Calibration failed: ${err.message}`);
    stopCamera();
    videoEl.style.display = "none";
    btnCal.disabled = false;
    setRing(0);
    return;
  }

  // ── step 5: persist calibration & redirect ────────────────────────────────
  state.setCalibration(
    calResult.node_id,
    calResult.anchor_alpha,
    calResult.heading_offset_deg ?? 0
  );
  state.setCurrentNode(calResult.node_id);

  const destNodeId = getDestNodeId();
  if (destNodeId && !state.getDestination()) {
    // Restore destination from URL param if state was cleared
    state.setDestination(destNodeId, destNodeId);
  }

  stopCamera();

  setStatus("Calibrated!", `You appear to be at: ${calResult.label}`);
  setRing(FRAME_COUNT);
  showBanner(
    `Located at ${calResult.label} (${Math.round(calResult.confidence * 100)}% confidence)`,
    "success"
  );

  // Brief display of success before redirecting
  await new Promise(r => setTimeout(r, 1200));

  const dest = destNodeId ?? state.getDestination()?.node_id;
  window.location.href = dest
    ? `ar.html?dest=${encodeURIComponent(dest)}`
    : "ar.html";
});
