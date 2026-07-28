"""
Flask API server for Vision Navigator
Augmented with: coordinates, waypoints, /api/calibrate, /api/zabgpt, in-memory image handling.
"""

import os
# Must be set before any ML library loads — prevents OMP duplicate-lib crash
# when PyTorch and sentence-transformers are loaded in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import io
import sys
import logging
from collections import Counter
from functools import wraps
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

BASE_DIR = Path(__file__).parent.absolute()
WEB_DIR  = BASE_DIR / "web"
sys.path.insert(0, str(BASE_DIR))

from app import CampusNavigationApp
from config import API_VERSION, CORS_ORIGINS
from pathfinding import NODE_COORDS, NODE_LABELS, CampusGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

flask_app = Flask(__name__)
flask_app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# Lazy singleton
navigation_app = None
_campus_graph = CampusGraph()


def get_navigation_app() -> CampusNavigationApp:
    global navigation_app
    if navigation_app is None:
        logger.info("Initialising CampusNavigationApp…")
        navigation_app = CampusNavigationApp()
        logger.info("CampusNavigationApp ready.")
    return navigation_app


def require_app(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            get_navigation_app()
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"App init error: {e}")
            return jsonify({"error": str(e)}), 500
    return wrapper


# ── helpers ───────────────────────────────────────────────────────────────────

def _coords(node_id: str) -> dict:
    c = NODE_COORDS.get(node_id, {"x": 0.0, "y": 0.0, "floor": 0})
    return {"x": c["x"], "y": c["y"], "floor": c["floor"]}


def _path_to_waypoints(path: list) -> list:
    return [
        {
            "node_id": n,
            "label":   NODE_LABELS.get(n, n),
            **_coords(n),
        }
        for n in path
    ]


def _pil_from_request_file(file_storage) -> Image.Image:
    """Read a werkzeug FileStorage into a PIL Image without touching disk."""
    data = file_storage.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


# ── endpoints ──────────────────────────────────────────────────────────────────

@flask_app.route("/api/health", methods=["GET"])
def health():
    app_ready = navigation_app is not None
    index_size = 0
    if app_ready:
        try:
            index_size = navigation_app.faiss_index.get_total_items()
        except Exception:
            pass
    return jsonify({
        "status":       "ok",
        "model_loaded": app_ready,
        "index_size":   index_size,
        "version":      API_VERSION,
    }), 200


@flask_app.route("/api/locations", methods=["GET"])
def get_locations():
    locations = [
        _campus_graph.get_node_info(node_id)
        for node_id in _campus_graph.graph
    ]
    return jsonify({"locations": locations, "count": len(locations)}), 200


@flask_app.route("/api/identify", methods=["POST"])
@require_app
def identify_location():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    use_voting = request.form.get("use_voting", "true").lower() == "true"

    try:
        pil_image = _pil_from_request_file(file)
        app = get_navigation_app()
        result = app.identify_from_pil(pil_image, use_voting=use_voting)
    except Exception as e:
        logger.error(f"identify error: {e}")
        return jsonify({"error": str(e)}), 500

    if "error" in result:
        return jsonify(result), 400

    node_id = result.get("location", "")
    result["node_id"]     = node_id
    result["coordinates"] = _coords(node_id)
    result["label"]       = NODE_LABELS.get(node_id, node_id)
    return jsonify(result), 200


@flask_app.route("/api/navigate", methods=["POST"])
@require_app
def navigate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Accept both old (start/end) and new (from_node/to_node) field names
    from_node = data.get("from_node") or data.get("start")
    to_node   = data.get("to_node")   or data.get("end")

    if not from_node or not to_node:
        return jsonify({"error": "Provide from_node and to_node"}), 400

    try:
        app = get_navigation_app()
        result = app.get_navigation_path(from_node, to_node)
    except Exception as e:
        logger.error(f"navigate error: {e}")
        return jsonify({"error": str(e)}), 500

    if "error" in result:
        return jsonify(result), 400

    result["waypoints"] = _path_to_waypoints(result.get("path", []))
    return jsonify(result), 200


@flask_app.route("/api/full-navigate", methods=["POST"])
@require_app
def full_navigate():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    if "destination" not in request.form:
        return jsonify({"error": "No destination provided"}), 400

    file        = request.files["image"]
    destination = request.form["destination"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        pil_image = _pil_from_request_file(file)
        app = get_navigation_app()

        loc_result = app.identify_from_pil(pil_image)
        if "error" in loc_result:
            return jsonify({"location_identification": loc_result}), 400

        current_node = loc_result.get("location", "")
        nav_result   = app.get_navigation_path(current_node, destination)

        waypoints = _path_to_waypoints(nav_result.get("path", []))

        return jsonify({
            "location_identification": {
                **loc_result,
                "node_id":     current_node,
                "coordinates": _coords(current_node),
                "label":       NODE_LABELS.get(current_node, current_node),
            },
            "navigation":          nav_result,
            "waypoints":           waypoints,
            "current_coordinates": _coords(current_node),
        }), 200

    except Exception as e:
        logger.error(f"full-navigate error: {e}")
        return jsonify({"error": str(e)}), 500


@flask_app.route("/api/calibrate", methods=["POST"])
@require_app
def calibrate():
    """
    Accept N≥3 frames (images[]) + matching alphas[] floats.
    Runs DINOv2+FAISS on each frame, picks majority node,
    returns anchor_alpha from the highest-confidence matching frame.
    """
    files  = request.files.getlist("images[]")
    alphas = request.form.getlist("alphas[]")

    if len(files) < 3:
        return jsonify({"error": "At least 3 frames required"}), 400
    if len(files) != len(alphas):
        return jsonify({"error": "images[] and alphas[] must have the same length"}), 400

    try:
        alphas = [float(a) for a in alphas]
    except ValueError:
        return jsonify({"error": "alphas[] must be floats"}), 400

    app = get_navigation_app()
    frame_results = []

    for idx, (file, alpha) in enumerate(zip(files, alphas)):
        try:
            pil_image = _pil_from_request_file(file)
            res = app.identify_from_pil(pil_image, use_voting=False)
            if "error" not in res:
                frame_results.append({
                    "node_id":    res["location"],
                    "confidence": res["confidence"],
                    "alpha":      alpha,
                })
        except Exception as e:
            logger.warning(f"Calibration frame {idx} failed: {e}")

    if not frame_results:
        return jsonify({"error": "All frames failed processing"}), 500

    # Majority vote across frames
    vote_counts   = Counter(r["node_id"] for r in frame_results)
    majority_node = vote_counts.most_common(1)[0][0]

    # Among frames that matched the majority node, pick highest confidence
    matching = [r for r in frame_results if r["node_id"] == majority_node]
    best     = max(matching, key=lambda r: r["confidence"])

    return jsonify({
        "node_id":            majority_node,
        "label":              NODE_LABELS.get(majority_node, majority_node),
        "anchor_alpha":       best["alpha"],
        "confidence":         best["confidence"],
        "heading_offset_deg": 0.0,
        "n_frames_used":      len(frame_results),
        "votes":              dict(vote_counts),
    }), 200


_zabgpt_nav = None  # cached singleton — avoids slow ChromaDB reload on every request

@flask_app.route("/api/zabgpt", methods=["POST"])
def zabgpt():
    """Proxy to ZabGPT RAG pipeline."""
    global _zabgpt_nav
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Provide {query}"}), 400

    try:
        if _zabgpt_nav is None:
            zabgpt_dir = BASE_DIR / "zabgpt"
            if str(zabgpt_dir) not in sys.path:
                sys.path.insert(0, str(zabgpt_dir))
            from dotenv import load_dotenv
            load_dotenv(zabgpt_dir / ".env", override=True)
            from src.rag import VisionNavigator  # type: ignore
            _zabgpt_nav = VisionNavigator()
            logger.info("ZabGPT VisionNavigator initialised and cached.")

        result = _zabgpt_nav.ask(data["query"])
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"ZabGPT error: {e}")
        _zabgpt_nav = None  # reset on error so next request retries init
        return jsonify({"error": str(e)}), 500


# ── CORS ───────────────────────────────────────────────────────────────────────

@flask_app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"]  = CORS_ORIGINS
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,ngrok-skip-browser-warning,X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@flask_app.route("/options-preflight", methods=["OPTIONS"])
def handle_options():
    return "", 204


# ── static web files ──────────────────────────────────────────────────────────

@flask_app.route("/")
def serve_index():
    return send_from_directory(WEB_DIR, "index.html")

@flask_app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(WEB_DIR, filename)


# ── error handlers ─────────────────────────────────────────────────────────────

@flask_app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "endpoints": [
        "GET  /api/health", "GET  /api/locations",
        "POST /api/identify", "POST /api/navigate",
        "POST /api/full-navigate", "POST /api/calibrate",
        "POST /api/zabgpt",
    ]}), 404


@flask_app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    cert = os.path.join(BASE_DIR, "cert.pem")
    key  = os.path.join(BASE_DIR, "key.pem")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    if os.path.exists(cert) and os.path.exists(key):
        logger.info("Vision Navigator API  v%s  -> https://0.0.0.0:%s", API_VERSION, port)
        flask_app.run(host="0.0.0.0", port=port, debug=False, threaded=True,
                      ssl_context=(cert, key))
    else:
        logger.info("Vision Navigator API  v%s  -> http://0.0.0.0:%s", API_VERSION, port)
        flask_app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
