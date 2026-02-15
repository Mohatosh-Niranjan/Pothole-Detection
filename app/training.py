import os
import glob
import shutil
from typing import Dict, List, Tuple

import yaml
from ultralytics import YOLO


def create_yolo_dataset_yaml(dataset_dir: str, out_yaml_path: str) -> str:
    # Check if it's the new potholeDataset structure
    annotated_images_dir = os.path.join(dataset_dir, "annotated-images")
    splits_file = os.path.join(dataset_dir, "splits.json")
    
    if os.path.isdir(annotated_images_dir) and os.path.exists(splits_file):
        # New potholeDataset structure
        import json
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        train_xmls = splits["train"]
        test_xmls = splits["test"]
        
        # Get corresponding image paths
        train_images = []
        test_images = []
        
        for xml_file in train_xmls:
            img_file = xml_file.replace('.xml', '.jpg')
            img_path = os.path.join(annotated_images_dir, img_file)
            if os.path.exists(img_path):
                train_images.append(img_path)
        
        for xml_file in test_xmls:
            img_file = xml_file.replace('.xml', '.jpg')
            img_path = os.path.join(annotated_images_dir, img_file)
            if os.path.exists(img_path):
                test_images.append(img_path)
        
        annotations_dir = annotated_images_dir
        print(f"Found {len(train_images)} training images and {len(test_images)} test images")
        
    else:
        # Old dataset structure
        images_dir = os.path.join(dataset_dir, "images")
        annotations_dir = os.path.join(dataset_dir, "annotations")
        if not os.path.isdir(images_dir) or not os.path.isdir(annotations_dir):
            raise RuntimeError("Expected dataset/images and dataset/annotations with VOC XML")

        # Split by filename convention or simple 80/20 split
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        n = len(image_paths)
        if n == 0:
            raise RuntimeError("No images found in dataset/images")
        split = max(1, int(0.8 * n))
        train_images = image_paths[:split]
        test_images = image_paths[split:]

    # Prepare a yolo-compatible folder
    yolo_root = os.path.join(dataset_dir, "yolo")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(yolo_root, sub), exist_ok=True)

    # Convert Pascal VOC XML to YOLO TXT labels
    from xml.etree import ElementTree as ET

    def voc_to_yolo_bbox(w: int, h: int, xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[float, float, float, float]:
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / float(w)
        bh = (ymax - ymin) / float(h)
        return x_center, y_center, bw, bh

    def convert_for_split(paths: List[str], split_name: str) -> None:
        for img_path in paths:
            file_stem = os.path.splitext(os.path.basename(img_path))[0]
            xml_path = os.path.join(annotations_dir, f"{file_stem}.xml")
            if not os.path.exists(xml_path):
                # Skip images without annotations
                continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find("size")
            w = int(size.findtext("width", default="0"))
            h = int(size.findtext("height", default="0"))
            label_lines: List[str] = []
            for obj in root.findall("object"):
                name = obj.findtext("name", default="pothole").strip().lower()
                if name != "pothole":
                    continue
                bnd = obj.find("bndbox")
                xmin = int(float(bnd.findtext("xmin", default="0")))
                ymin = int(float(bnd.findtext("ymin", default="0")))
                xmax = int(float(bnd.findtext("xmax", default="0")))
                ymax = int(float(bnd.findtext("ymax", default="0")))
                x, y, bw, bh = voc_to_yolo_bbox(w, h, xmin, ymin, xmax, ymax)
                label_lines.append(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            # Copy image and write label file
            dst_img = os.path.join(yolo_root, f"images/{split_name}/{os.path.basename(img_path)}")
            shutil.copy2(img_path, dst_img)
            dst_lbl = os.path.join(yolo_root, f"labels/{split_name}/{file_stem}.txt")
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    convert_for_split(train_images, "train")
    convert_for_split(test_images, "val")  # YOLO uses 'val' for validation

    data_yaml = {
        "path": yolo_root,
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pothole"},
        "nc": 1,
    }
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f)
    return out_yaml_path


def train_pothole_model(data_yaml: str, model: str = "yolov8n.pt", epochs: int = 50, imgsz: int = 640, device: str = "") -> str:
    yolo = YOLO(model)
    results = yolo.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device)
    # The best model path is typically runs/detect/train*/weights/best.pt
    best_path = None
    for run_dir in sorted(glob.glob("runs/detect/train*")):
        cand = os.path.join(run_dir, "weights", "best.pt")
        if os.path.exists(cand):
            best_path = cand
    if best_path is None:
        raise RuntimeError("Training finished but best.pt not found")
    return best_path


