import os
import sys
from typing import Optional

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.detection import PotholeDetector, classify_risk
from app.realtime_store import insert_detection


st.set_page_config(page_title="PotholeTrack Live Stream", layout="wide")
st.title("📡 Live Pothole Detection (WebRTC)")
st.caption("Use your device camera to stream video and detect potholes in real-time.")

weights = st.sidebar.text_input("YOLO weights", value="outputs/best.pt")
conf = st.sidebar.slider("Confidence", 0.05, 0.9, 0.25, 0.05)
iou = st.sidebar.slider("IoU", 0.1, 0.9, 0.5, 0.05)

detector = PotholeDetector(weights_path=weights, conf_threshold=conf, iou_threshold=iou)


class Processor(VideoProcessorBase):
    def __init__(self) -> None:
        self.frame_id = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = detector.model(img, conf=detector.conf_threshold, iou=detector.iou_threshold, verbose=False)
        annotated = img.copy()
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
                cv2 = __import__("cv2")
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                label = f"{risk} {confidence:.2f}"
                cv2.putText(annotated, label, (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                # Store detection with NaN GPS (to be correlated by device_id if posted separately)
                # In a real deployment, augment with device GPS via the GPS API
                try:
                    insert_detection(
                        __import__("app.detection", fromlist=["DetectionRecord"]).detection.DetectionRecord(
                            frame_id=self.frame_id,
                            bbox_x1=float(x1), bbox_y1=float(y1), bbox_x2=float(x2), bbox_y2=float(y2),
                            confidence=confidence, area_px=area_px, risk_level=risk,
                            latitude=float("nan"), longitude=float("nan"), detection_time=""
                        )
                    )
                except Exception:
                    pass
        self.frame_id += 1
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


webrtc_streamer(
    key="pothole-live",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)


