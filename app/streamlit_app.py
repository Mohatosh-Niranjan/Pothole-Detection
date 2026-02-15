import os
import io
import sys
import base64
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import av
import streamlit.components.v1 as components

# Ensure project root (parent of this file's directory) is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detection import PotholeDetector, classify_risk, DetectionRecord
from app.heatmap import build_map
from app.gps import parse_gpx_file, resample_track_to_length
from app.realtime_store import insert_detection, get_recent_detections, get_latest_gps
from app.gps_api import app as gps_fastapi

APP_TITLE = "PotholeTrack — Real-time Pothole Detection"
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


def ensure_outputs_dir() -> None:
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR, exist_ok=True)


def save_csv(df: pd.DataFrame, stem: str) -> str:
    ensure_outputs_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUTPUTS_DIR, f"{stem}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def save_map_html(m, stem: str) -> str:
    ensure_outputs_dir()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(OUTPUTS_DIR, f"{stem}_{ts}.html")
    m.save(path)
    return path


def render_detection_outputs(df: pd.DataFrame, annotated_frames, saved_video_path: Optional[str], context: str = "") -> None:
    if df.empty:
        st.info("No potholes detected in the video.")
        return

    if context:
        st.markdown(f"### 📊 Results {context}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(df))
    with col2:
        st.metric("High Risk", int((df["risk_level"] == "High").sum()))
    with col3:
        st.metric("Unique Frames", df["frame_id"].nunique())

    if saved_video_path and os.path.exists(saved_video_path):
        st.subheader("🎥 Annotated Video")
        video_bytes = None
        try:
            st.video(saved_video_path)
        except Exception:
            try:
                with open(saved_video_path, "rb") as vf:
                    video_bytes = vf.read()
                st.video(video_bytes)
            except Exception as e:
                st.warning(f"Video preview not available: {e}")
                st.info(f"You can download the video here: {saved_video_path}")

        if video_bytes is None and os.path.exists(saved_video_path):
            with open(saved_video_path, "rb") as vf:
                video_bytes = vf.read()
        if video_bytes:
            st.download_button(
                "📥 Download Annotated Video",
                video_bytes,
                file_name=os.path.basename(saved_video_path),
                mime="video/mp4",
            )

    if annotated_frames:
        st.subheader("🖼️ Sample Detection Frames")
        cols = st.columns(3)
        for idx, frame in enumerate(annotated_frames[:6]):
            with cols[idx % 3]:
                st.image(frame[:, :, ::-1], channels="RGB", caption=f"Frame {idx}")

    st.subheader("📋 Detection Details")
    st.dataframe(df)
    csv_path = save_csv(df, stem="pothole_detections")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "📥 Download CSV Data",
        csv_buf.getvalue(),
        file_name=os.path.basename(csv_path),
        mime="text/csv",
    )

    st.subheader("🗺️ Pothole Location Map")
    m = build_map(df)
    st_folium(m, width=None, height=600, key=f"map_{len(df)}_{context}")
    try:
        components.html(m._repr_html_(), height=600)
    except Exception:
        pass
    map_path = save_map_html(m, stem="pothole_map")
    st.success(f"Map saved: {map_path}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title("🚧 PotholeTrack")
    st.markdown("**Real-time Pothole Detection & Risk Assessment**")
    
    # Initialize session state
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'map_created' not in st.session_state:
        st.session_state.map_created = False
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["📹 Video Detection", "📊 Results & Map", "📡 Live Feed"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload Video")
            
            uploaded = st.file_uploader(
                "Upload your video file",
                type=["mp4", "mov", "avi", "mkv"],
                help="Upload a video file to detect potholes"
            )
        
        with col2:
            st.subheader("⚙️ Detection Settings")
            
            # Model selection - prioritize trained model
            trained_model_path = os.path.join(OUTPUTS_DIR, "best.pt")
            if os.path.exists(trained_model_path):
                st.success("✅ Found your trained pothole model!")
                weights = trained_model_path
                st.info(f"Using: {os.path.basename(trained_model_path)}")
            else:
                st.warning("⚠️ No trained model found. Using default YOLOv8 (won't detect potholes)")
                st.info("💡 To detect potholes, you need to train a custom model first. Check the training section below.")
                weights = "yolov8n.pt"
                
            # Option to upload different model
            uploaded_weights = st.file_uploader("Or upload different .pt model", type=["pt"], key="weights")
            if uploaded_weights is not None:
                ensure_outputs_dir()
                weights = os.path.join(OUTPUTS_DIR, uploaded_weights.name)
                if not os.path.exists(weights):
                    with open(weights, "wb") as wf:
                        wf.write(uploaded_weights.read())
                st.success(f"Using uploaded model: {uploaded_weights.name}")
            
            # Detection parameters
            conf_thres = st.slider("Confidence", 0.05, 0.9, 0.25, 0.05, help="Lower = more detections (try 0.1-0.3 for better detection)")
            iou_thres = st.slider("IoU Threshold", 0.1, 0.9, 0.5, 0.05, help="Non-maximum suppression threshold")
            
            # Additional options
            st.markdown("**🔧 Advanced Options**")
            show_all_classes = st.checkbox("Show all detected objects (not just potholes)", value=False, help="If no potholes detected, show all objects the model finds")
            
            # GPS settings
            st.markdown("**📍 Location Settings**")
            gps_mode = st.radio("GPS Source:", ["Simulated Path", "Fixed Location", "GPX File"])
            
            if gps_mode == "Fixed Location":
                base_lat = st.number_input("Latitude", value=28.6139, format="%0.6f")
                base_lon = st.number_input("Longitude", value=77.2090, format="%0.6f")
                gpx_file = None
            elif gps_mode == "GPX File":
                gpx_file = st.file_uploader("Upload GPX file", type=["gpx"], key="gpx")
                base_lat, base_lon = 28.6139, 77.2090
            else:
                base_lat = st.number_input("Start Latitude", value=28.6139, format="%0.6f")
                base_lon = st.number_input("Start Longitude", value=77.2090, format="%0.6f")
                gpx_file = None

        # Detection button and processing
        if uploaded is not None and weights is not None:
            if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                with st.spinner("🔍 Running pothole detection..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def on_progress(done: int, total: int) -> None:
                        if total > 0:
                            progress_bar.progress(min(1.0, done / total))
                            status_text.text(f"Processing frame {done}/{total}")

                    # Prepare GPS data
                    gps_points = None
                    gps_times_iso = None
                    if gps_mode == "GPX file" and gpx_file is not None:
                        gpx_bytes = gpx_file.read()
                        points, times = parse_gpx_file(gpx_bytes)
                        gps_points = points
                        gps_times_iso = [t.isoformat() + "Z" for t in times]
                    elif gps_mode == "Fixed Location":
                        gps_points = [(base_lat, base_lon)]
                        gps_times_iso = None

                    # Process video
                    ensure_outputs_dir()
                    tmp_video_path = os.path.join(OUTPUTS_DIR, f"uploaded_{uploaded.name}")
                    with open(tmp_video_path, "wb") as f:
                        f.write(uploaded.read())
                    
                    output_video_path = os.path.join(OUTPUTS_DIR, f"annotated_{uploaded.name}")
                    
                    detector = PotholeDetector(weights_path=weights, conf_threshold=conf_thres, iou_threshold=iou_thres)
                    
                    # Debug: Show model info
                    st.info(f"Using model: {weights}")
                    st.info(f"Confidence threshold: {conf_thres}, IoU threshold: {iou_thres}")
                    
                    df, annotated_frames, saved_video_path = detector.process_video(
                        tmp_video_path,
                        base_latitude=base_lat,
                        base_longitude=base_lon,
                        simulate_gps=(gps_mode == "Simulated Path"),
                        gps_points=gps_points,
                        gps_times=gps_times_iso,
                        max_frames=None,
                        sample_stride=1,
                        progress_callback=on_progress,
                        save_annotated_video=True,
                        output_video_path=output_video_path,
                        show_all_classes=show_all_classes,
                        detection_sink=insert_detection,
                    )
                    
                    # Debug: Show detection results
                    st.info(f"Detection completed. Found {len(df)} detections in {len(annotated_frames)} frames")
                    
                    # Store results in session state
                    st.session_state.detection_results = {
                        'df': df,
                        'annotated_frames': annotated_frames,
                        'saved_video_path': saved_video_path,
                        'map_created': False
                    }
                    
                    status_text.text("✅ Detection complete!")
                    st.success("Detection completed! See below and the Results & Map tab.")

                    with st.expander("View results without leaving this tab", expanded=True):
                        render_detection_outputs(df, annotated_frames, saved_video_path, context="(latest run)")
        
        elif uploaded is None:
            st.info("👆 Please upload a video file to start detection")
        elif weights is None:
            st.warning("⚠️ Please upload a model file or select default model")

    with tab2:
        st.subheader("📊 Detection Results")
        
        if st.session_state.detection_results is not None:
            df = st.session_state.detection_results['df']
            annotated_frames = st.session_state.detection_results['annotated_frames']
            saved_video_path = st.session_state.detection_results['saved_video_path']
            render_detection_outputs(df, annotated_frames, saved_video_path, context="(from history)")
            st.session_state.map_created = True
        else:
            st.info("👆 Please run detection first in the Video Detection tab")

    # Live camera tab
    with tab3:
        st.subheader("📡 Live Camera Detection")
        st.caption("Use your device camera to stream frames and detect potholes in real time. Detections are stored and shown on the live map below.")

        # Live model settings
        live_col1, live_col2 = st.columns([2, 1])
        with live_col2:
            live_weights = st.text_input("YOLO weights", value=os.path.join(OUTPUTS_DIR, "best.pt"))
            live_conf = st.slider("Confidence", 0.05, 0.9, 0.25, 0.05, key="live_conf")
            live_iou = st.slider("IoU", 0.1, 0.9, 0.5, 0.05, key="live_iou")
            device_id = st.text_input("Device ID", value="device-live")
            gps_api_url = st.text_input("GPS API URL", value="http://localhost:8000/gps", help="Public URL of your GPS API (needs HTTPS when on phone).")
            fallback_lat = st.number_input("Fallback Latitude", value=28.6139, format="%0.6f")
            fallback_lon = st.number_input("Fallback Longitude", value=77.2090, format="%0.6f")

        st.info("Allow camera + location. GPS updates stream automatically to the API above.")
        if gps_api_url:
            components.html(
                f"""
                <script>
                (function() {{
                    const apiUrl = "{gps_api_url}";
                    const deviceId = "{device_id}";
                    if (!navigator.geolocation) {{
                        console.log("Geolocation unsupported.");
                        return;
                    }}
                    navigator.geolocation.watchPosition(function(pos) {{
                        fetch(apiUrl, {{
                            method: "POST",
                            headers: {{ "Content-Type": "application/json" }},
                            body: JSON.stringify({{
                                device_id: deviceId,
                                latitude: pos.coords.latitude,
                                longitude: pos.coords.longitude,
                                timestamp: new Date().toISOString()
                            }})
                        }}).catch(err => console.log("GPS post failed", err));
                    }}, function(err) {{
                        console.log("GPS error", err);
                    }}, {{
                        enableHighAccuracy: true,
                        maximumAge: 5000,
                        timeout: 10000
                    }});
                }})();
                </script>
                """,
                height=0,
            )

        # Lazy-initialize detector for live mode (inside processor closure)
        detector_holder = {"detector": None}
        live_config = {
            "device_id": device_id,
            "fallback_lat": fallback_lat,
            "fallback_lon": fallback_lon,
        }

        class LiveProcessor(VideoProcessorBase):
            def __init__(self) -> None:
                self.frame_id = 0

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                import cv2
                if detector_holder["detector"] is None:
                    detector_holder["detector"] = PotholeDetector(
                        weights_path=live_weights or "yolov8n.pt",
                        conf_threshold=live_conf,
                        iou_threshold=live_iou,
                    )
                detector = detector_holder["detector"]

                img = frame.to_ndarray(format="bgr24")
                results = detector.model(img, conf=detector.conf_threshold, iou=detector.iou_threshold, verbose=False)
                annotated = img.copy()

                latest_latlon = None
                if live_config["device_id"]:
                    latest_latlon = get_latest_gps(live_config["device_id"])

                lat = lon = None
                if latest_latlon:
                    lat, lon, _ = latest_latlon
                else:
                    lat = live_config["fallback_lat"]
                    lon = live_config["fallback_lon"]

                for result in results:
                    if not hasattr(result, "boxes") or result.boxes is None:
                        continue
                    boxes = result.boxes
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
                    if xyxy is None or confs is None:
                        continue
                    for j, (x1, y1, x2, y2) in enumerate(xyxy):
                        confidence = float(confs[j])
                        area_px = float(max(0, x2 - x1) * max(0, y2 - y1))
                        risk = classify_risk(area_px, confidence)
                        color = (0, 255, 255) if risk == "Low" else (0, 165, 255) if risk == "Medium" else (0, 0, 255)
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                        label = f"{risk} {confidence:.2f}"
                        cv2.putText(annotated, label, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                        try:
                            insert_detection(
                                DetectionRecord(
                                    frame_id=self.frame_id,
                                    bbox_x1=float(x1), bbox_y1=float(y1), bbox_x2=float(x2), bbox_y2=float(y2),
                                    confidence=confidence, area_px=area_px, risk_level=risk,
                                    latitude=float(lat), longitude=float(lon), detection_time=datetime.utcnow().isoformat()+"Z",
                                    device_id=live_config["device_id"],
                                )
                            )
                        except Exception:
                            pass
                self.frame_id += 1
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        with live_col1:
            webrtc_streamer(
                key="pothole-live-tab",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=LiveProcessor,
                media_stream_constraints={"video": True, "audio": False},
            )

        # Live map (auto-refresh)
        st.subheader("🗺️ Live Map")
        detections = get_recent_detections(limit=1000)
        live_df = pd.DataFrame(detections)
        if live_df.empty:
            st.info("Waiting for live detections...")
        else:
            live_map = build_map(live_df)
            st_folium(live_map, width=None, height=600, key=f"live_map_{len(live_df)}")
        st.button("Refresh live map")

    # Training section at the bottom
    st.markdown("---")
    with st.expander("🔧 Train Your Own Pothole Model (Advanced)"):
        st.markdown("**To detect potholes, you need a custom-trained model.**")
        
        if st.button("🚀 Train Pothole Model from potholeDataset", type="secondary"):
            with st.spinner("Training custom pothole model..."):
                try:
                    from app.training import create_yolo_dataset_yaml, train_pothole_model
                    
                    dataset_dir = os.path.join(PROJECT_ROOT, "potholeDataset")
                    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
                    yaml_path = create_yolo_dataset_yaml(dataset_dir, data_yaml_path)
                    best_model = train_pothole_model(yaml_path, model="yolov8n.pt", epochs=30, imgsz=640)
                    
                    # Copy to outputs
                    import shutil
                    shutil.copy2(best_model, os.path.join(OUTPUTS_DIR, "best.pt"))
                    
                    st.success(f"✅ Training complete! Model saved: {best_model}")
                    st.info("🔄 Refresh the page to use your new model")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")


if __name__ == "__main__":
    main()