# PotholeTrack — Real-time Pothole Detection

PotholeTrack is a production-oriented toolkit for detecting, mapping, and monitoring potholes from video and live camera feeds. It combines a YOLOv8-based detector, GPS correlation, realtime storage, and interactive maps to provide an end-to-end pipeline for field data collection, visualization, and model training.

**Why it matters**
- **Make roads safer:** Automatically locate and prioritize potholes for repair using objective visual evidence.
- **Scales from a phone to a fleet:** Supports live WebRTC camera streams (phone or laptop) and batch video uploads.
- **Actionable output:** Generates GPS-tagged detections, heatmaps, CSV reports, and annotated videos.

**Key Features**
- **Dual input modes:** Live camera (Streamlit WebRTC) and video file processing.
- **YOLOv8-powered detection:** Pluggable weights; defaults to `yolov8n.pt` or your trained `runs\detect\train3\weights\best.pt`.
- **GPS support:** Simulated track, fixed-point fallback, or GPX uploads; device GPS ingested via a small FastAPI endpoint.
- **Visualizations:** Folium maps + heatmaps and Streamlit dashboards for monitoring and export.
- **Training helper:** Convert VOC/XML annotations into YOLO format and run training using Ultralytics.


**Architecture & Data Flow (short)**
- Video / Camera → `PotholeDetector.process_video()` → detection records (`DetectionRecord`) → saved CSVs + annotated video → `app/realtime_store.insert_detection()` writes to `outputs/realtime.db` → `app/heatmap.build_map()` produces folium maps → UI (Streamlit / Flask) serves maps and reports.

Quickstart (Windows)
1. Create and activate a venv (recommended):

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the GPS API (for live device GPS pings):

```powershell
uvicorn app.gps_api:app --host 0.0.0.0 --port 8000
```

3. Run the Streamlit app (interactive detection & live):

```powershell
streamlit run app/streamlit_app.py
```

4. Run the Flask upload/results UI (video upload + map export):

```powershell
python app/flask_app.py
```

5. (Optional) Run the dashboard or live stream pages separately:

```powershell
streamlit run app/dashboard_app.py
streamlit run app/live_stream.py
```

Training (outline)
- Prepare your dataset under `dataset/` either as VOC XMLs in `dataset/annotations` + images in `dataset/images` or use the `potholeDataset` layout. Use `app/training.create_yolo_dataset_yaml()` to generate a YOLO `data.yaml`.
- Train with Ultralytics (example API call from Python):

```python
from app.training import create_yolo_dataset_yaml, train_pothole_model
create_yolo_dataset_yaml('dataset', 'dataset/data.yaml')
train_pothole_model('dataset/data.yaml', model='yolov8n.pt', epochs=50)
```

Outputs
- Trained model: `outputs/best.pt` (or `runs/detect/.../weights/best.pt`)
- CSV exports: `outputs/pothole_detections_*.csv`
- HTML maps: `outputs/pothole_map_*.html`
- Realtime DB: `outputs/realtime.db`

Notes & Recommendations
- Pin dependency versions for reproducibility; `requirements.txt` currently lists packages without versions.
- GPU: If you have CUDA, configure the `device` when constructing `PotholeDetector` and ensure `ultralytics` sees the GPU.
- Model labels: The detector prefers a `pothole` class name; otherwise it falls back to road-related classes or all classes.


