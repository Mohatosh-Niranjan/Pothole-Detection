import os
import sys
from datetime import datetime
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detection import PotholeDetector
from app.heatmap import build_map
from app.gps import parse_gpx_file
from app.realtime_store import insert_detection, get_recent_detections

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = OUTPUTS_DIR


def save_csv(df: pd.DataFrame, stem: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUTPUTS_DIR, f"{stem}_{ts}.csv")
    df.to_csv(path, index=False)
    return os.path.basename(path)


def save_map_html(df: pd.DataFrame, stem: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUTPUTS_DIR, f"{stem}_{ts}.html")
    m = build_map(df)
    m.save(path)
    return os.path.basename(path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files.get("video")
        if not file or file.filename == "":
            return render_template("upload.html", error="Please select a video file.")

        # Save uploaded file
        video_name = f"uploaded_{file.filename}"
        video_path = os.path.join(OUTPUTS_DIR, video_name)
        file.save(video_path)

        # Optional GPX
        gpx_file = request.files.get("gpx")
        gps_points = None
        gps_times_iso = None
        if gpx_file and gpx_file.filename.endswith(".gpx"):
            gpx_bytes = gpx_file.read()
            points, times = parse_gpx_file(gpx_bytes)
            gps_points = points
            gps_times_iso = [t.isoformat() + "Z" for t in times]

        # Detection parameters
        conf = float(request.form.get("conf", 0.25))
        iou = float(request.form.get("iou", 0.45))
        base_lat = float(request.form.get("base_lat", 28.6139))
        base_lon = float(request.form.get("base_lon", 77.2090))

        weights = request.form.get("weights", "outputs/best.pt").strip() or "yolov8n.pt"

        detector = PotholeDetector(weights_path=weights, conf_threshold=conf, iou_threshold=iou)

        def sink(rec):
            try:
                insert_detection(rec)
            except Exception:
                pass

        out_video_name = f"annotated_{file.filename}"
        out_video_path = os.path.join(OUTPUTS_DIR, out_video_name)

        df, _, saved_video_path = detector.process_video(
            video_path,
            base_latitude=base_lat,
            base_longitude=base_lon,
            simulate_gps=(gps_points is None),
            gps_points=gps_points,
            gps_times=gps_times_iso,
            max_frames=None,
            sample_stride=1,
            progress_callback=None,
            save_annotated_video=True,
            output_video_path=out_video_path,
            show_all_classes=False,
            detection_sink=sink,
        )

        if df.empty:
            return render_template("upload.html", message="No potholes detected in the video.")

        csv_filename = save_csv(df, "pothole_detections")
        map_filename = save_map_html(df, "pothole_map")

        return render_template(
            "results.html",
            video_filename=os.path.basename(saved_video_path),
            csv_filename=csv_filename,
            map_filename=map_filename,
            records=df.to_dict(orient="records"),
        )

    return render_template("upload.html")


@app.route("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=False)


@app.route("/live")
def live():
    # This page only handles browser-side camera; detections and GPS go through existing APIs.
    return render_template("live.html")


@app.route("/api/live_map")
def api_live_map():
    detections = get_recent_detections(limit=1000)
    return jsonify(detections)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


