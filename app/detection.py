from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from .gps import generate_gps_track, resample_track_to_length


@dataclass
class DetectionRecord:
    frame_id: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    confidence: float
    area_px: float
    risk_level: str
    latitude: float
    longitude: float
    detection_time: str
    device_id: Optional[str] = None


def classify_risk(area_px: float, confidence: float) -> str:
    # Simple rule-based risk classification using area and confidence.
    # Tunable thresholds; larger areas and higher confidence => higher risk.
    if area_px >= 50_000 or (area_px >= 30_000 and confidence >= 0.6):
        return "High"
    if area_px >= 12_000 or (area_px >= 8_000 and confidence >= 0.5):
        return "Medium"
    return "Low"


class PotholeDetector:
    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        if device is not None:
            # The ultralytics model uses .to(device) for device selection
            self.model.to(device)

    def _extract_pothole_indices(self, class_names) -> Optional[List[int]]:
        """Return indices of 'pothole' classes if present; otherwise None.

        Supports Ultralytics formats:
        - dict: {class_id: class_name}
        - list/tuple: [class_name0, class_name1, ...]
        """
        try:
            if isinstance(class_names, dict):
                # Look for pothole class
                pothole_indices = [int(k) for k, v in class_names.items() if str(v).lower() == "pothole"]
                if pothole_indices:
                    return pothole_indices
                
                # Fallback: look for road-related objects if no pothole class
                road_related = ['road', 'street', 'pavement', 'asphalt', 'surface']
                fallback_indices = []
                for k, v in class_names.items():
                    if any(road_word in str(v).lower() for road_word in road_related):
                        fallback_indices.append(int(k))
                if fallback_indices:
                    print(f"Using fallback road-related classes: {[class_names[i] for i in fallback_indices]}")
                    return fallback_indices
                    
            if isinstance(class_names, (list, tuple)):
                # Look for pothole class
                pothole_indices = [i for i, v in enumerate(class_names) if str(v).lower() == "pothole"]
                if pothole_indices:
                    return pothole_indices
                    
                # Fallback: look for road-related objects
                road_related = ['road', 'street', 'pavement', 'asphalt', 'surface']
                fallback_indices = []
                for i, v in enumerate(class_names):
                    if any(road_word in str(v).lower() for road_word in road_related):
                        fallback_indices.append(i)
                if fallback_indices:
                    print(f"Using fallback road-related classes: {[class_names[i] for i in fallback_indices]}")
                    return fallback_indices
        except Exception:
            pass
        # Final fallback: keep all classes
        print("No pothole or road-related classes found, using all classes")
        return None

    def process_video(
        self,
        video_path: str,
        base_latitude: float = 28.6139,
        base_longitude: float = 77.2090,
        simulate_gps: bool = True,
        gps_points: Optional[List[Tuple[float, float]]] = None,
        gps_times: Optional[List[str]] = None,
        max_frames: Optional[int] = None,
        sample_stride: int = 1,
        progress_callback: Optional[callable] = None,
        save_annotated_video: bool = True,
        output_video_path: Optional[str] = None,
        show_all_classes: bool = False,
        detection_sink: Optional[callable] = None,
    ) -> Tuple[pd.DataFrame, List[np.ndarray], Optional[str]]:
        """Run detection on a video and return detections dataframe, annotated frames, and video path.

        The dataframe has columns:
        [frame_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, confidence, area_px,
         risk_level, latitude, longitude, detection_time]
        
        Returns:
            df: DataFrame with detections
            annotated_frames: List of annotated frame arrays
            output_video_path: Path to saved annotated video (if save_annotated_video=True)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        effective_frames = list(range(0, total_frames, sample_stride))
        if max_frames is not None:
            effective_frames = effective_frames[: max(1, max_frames)]

        # Determine GPS sequence for frames
        if gps_points is not None and len(gps_points) > 0:
            gps_track = resample_track_to_length(gps_points, len(effective_frames))
        elif simulate_gps:
            gps_track = generate_gps_track(
                num_points=len(effective_frames),
                base_latitude=base_latitude,
                base_longitude=base_longitude,
                bearing_degrees=0.0,
            )
        else:
            gps_track = [(np.nan, np.nan)] * len(effective_frames)

        class_names = self.model.names if hasattr(self.model, "names") else []
        pothole_class_indices = self._extract_pothole_indices(list(class_names))

        records: List[DetectionRecord] = []
        annotated_frames: List[np.ndarray] = []

        # Setup video writer if saving annotated video
        video_writer = None
        if save_annotated_video:
            if output_video_path is None:
                output_video_path = video_path.replace('.mp4', '_annotated.mp4').replace('.avi', '_annotated.avi').replace('.mov', '_annotated.mov')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        now_iso = datetime.utcnow().isoformat() + "Z"

        for i, frame_id in enumerate(effective_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue

            # Inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
            
            # Debug: Print detection info for first few frames
            if i < 3:
                print(f"Frame {i}: Processing frame {frame_id}")
                print(f"Model classes: {self.model.names}")
                print(f"Pothole class indices: {pothole_class_indices}")
                for result in results:
                    if hasattr(result, "boxes") and result.boxes is not None:
                        boxes = result.boxes
                        print(f"Found {len(boxes)} detections")
                        if len(boxes) > 0:
                            print(f"Confidences: {boxes.conf.cpu().numpy()}")
                            print(f"Classes: {boxes.cls.cpu().numpy()}")
                    else:
                        print("No detections in this frame")

            # Draw annotations
            annotated = frame.copy()

            lat, lon = gps_track[i]
            # Handle multiple prediction objects possibly returned
            for result in results:
                if not hasattr(result, "boxes") or result.boxes is None:
                    continue
                boxes = result.boxes
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
                conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
                cls = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else None
                if xyxy is None or conf is None:
                    continue

                for j, (x1, y1, x2, y2) in enumerate(xyxy):
                    confidence = float(conf[j])
                    class_id = int(cls[j]) if cls is not None else -1
                    
                    # Filter by class if not showing all classes
                    if not show_all_classes and pothole_class_indices is not None and class_id not in pothole_class_indices:
                        continue

                    bbox_w = max(0.0, float(x2 - x1))
                    bbox_h = max(0.0, float(y2 - y1))
                    area_px = float(bbox_w * bbox_h)
                    risk = classify_risk(area_px, confidence)

                    # Draw bounding box
                    color = (0, 255, 255) if risk == "Low" else (0, 165, 255) if risk == "Medium" else (0, 0, 255)
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Draw label with background
                    label = f"{risk} {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    label_y = max(0, int(y1) - 10)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(annotated, 
                                (int(x1), label_y - label_size[1] - 10), 
                                (int(x1) + label_size[0] + 10, label_y + 5), 
                                color, -1)
                    
                    # Draw text
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1) + 5, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),  # White text
                        2,
                        cv2.LINE_AA,
                    )

                    # Determine detection time
                    det_time = (
                        gps_times[i]
                        if (gps_times is not None and i < len(gps_times))
                        else now_iso
                    )

                    rec = DetectionRecord(
                            frame_id=frame_id,
                            bbox_x1=float(x1),
                            bbox_y1=float(y1),
                            bbox_x2=float(x2),
                            bbox_y2=float(y2),
                            confidence=confidence,
                            area_px=area_px,
                            risk_level=risk,
                            latitude=float(lat),
                            longitude=float(lon),
                            detection_time=det_time,
                        )
                    records.append(rec)
                    if detection_sink is not None:
                        try:
                            detection_sink(rec)
                        except Exception:
                            pass

            annotated_frames.append(annotated)
            
            # Write frame to video if saving
            if video_writer is not None:
                video_writer.write(annotated)

            if progress_callback is not None:
                progress_callback(i + 1, len(effective_frames))

        cap.release()
        if video_writer is not None:
            video_writer.release()

        df = pd.DataFrame([r.__dict__ for r in records])
        # Ensure consistent column order
        if not df.empty:
            df = df[
                [
                    "frame_id",
                    "bbox_x1",
                    "bbox_y1",
                    "bbox_x2",
                    "bbox_y2",
                    "confidence",
                    "area_px",
                    "risk_level",
                    "latitude",
                    "longitude",
                    "detection_time",
                ]
            ]

        return df, annotated_frames, output_video_path

