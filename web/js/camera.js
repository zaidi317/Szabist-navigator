/**
 * camera.js — rear camera access and JPEG frame capture.
 *
 * Rules:
 *   - startCamera() MUST be called inside a user gesture handler (click/tap).
 *   - Uses `ideal` (not `exact`) constraints so we never hard-fail on device quirks.
 *   - captureJpeg() caps output at 1280×720 and JPEG quality 0.7 (≤ ~2 MB).
 */

let _stream = null;

/**
 * Start the rear camera and attach it to a <video> element.
 * Returns the MediaStream, or throws if permission is denied.
 * @param {HTMLVideoElement} videoEl
 * @returns {Promise<MediaStream>}
 */
export async function startCamera(videoEl) {
  if (_stream) stopCamera();

  _stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: "environment" },
      width:      { ideal: 1280 },
      height:     { ideal: 720 },
    },
    audio: false,
  });

  videoEl.srcObject = _stream;

  // Wait until the video is actually playing before resolving
  await new Promise((resolve, reject) => {
    videoEl.onloadedmetadata = () => {
      videoEl.play().then(resolve).catch(reject);
    };
    videoEl.onerror = reject;
  });

  return _stream;
}

/**
 * Stop the camera and release the stream.
 */
export function stopCamera() {
  if (_stream) {
    _stream.getTracks().forEach(t => t.stop());
    _stream = null;
  }
}

/**
 * Capture one frame from the live video as a JPEG Blob.
 * Output is capped at 1280×720. Returns null if the video is not ready.
 * @param {HTMLVideoElement} videoEl
 * @param {number} quality  JPEG quality 0–1, default 0.7
 * @returns {Promise<Blob|null>}
 */
export function captureJpeg(videoEl, quality = 0.7) {
  if (!videoEl.videoWidth || !videoEl.videoHeight) return Promise.resolve(null);

  const MAX_W = 1280;
  const MAX_H = 720;
  const scale = Math.min(1, MAX_W / videoEl.videoWidth, MAX_H / videoEl.videoHeight);

  const canvas = document.createElement("canvas");
  canvas.width  = Math.round(videoEl.videoWidth  * scale);
  canvas.height = Math.round(videoEl.videoHeight * scale);

  canvas.getContext("2d").drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  return new Promise(resolve =>
    canvas.toBlob(blob => resolve(blob), "image/jpeg", quality)
  );
}

/**
 * Returns true if the browser reports camera permission as granted.
 * Falls back gracefully when the Permissions API is unavailable.
 * @returns {Promise<boolean>}
 */
export async function isCameraPermissionGranted() {
  try {
    const status = await navigator.permissions.query({ name: "camera" });
    return status.state === "granted";
  } catch {
    return false;
  }
}
