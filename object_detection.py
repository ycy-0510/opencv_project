"""Simple object detection wrapper for this project.

This module provides an ObjectDetection class with a detect(frame)
method that returns (class_ids, scores, boxes). It attempts to load
YOLO (if cfg+weights are present) and otherwise falls back to the
Haar cascade `face.xml` included in the repo.

The fallback ensures existing scripts (for example `obj_tracking.py`)
can run without the original `object_detection.py` from the web.
"""
from __future__ import annotations

import os
from typing import List, Tuple

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


class ObjectDetection:
    """Small adapter that exposes detect(frame) -> (class_ids, scores, boxes).

    - If YOLO config+weights are found in the same folder as this file
      (common names are checked), it will use OpenCV DNN to run YOLO.
    - Otherwise it loads the `face.xml` Haar cascade shipped in the repo
      and returns face detections as boxes.
    """

    def __init__(self, model_dir: str | None = None, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.labels = None
        self.mode = "cascade"

        base_dir = model_dir or os.path.dirname(__file__)

        # Try common YOLO file names
        candidates = [
            ("yolov4.weights", "yolov4.cfg"),
            ("yolov4-tiny.weights", "yolov4-tiny.cfg"),
            ("yolov3.weights", "yolov3.cfg"),
            ("yolov3-tiny.weights", "yolov3-tiny.cfg"),
        ]

        found = False
        for weights_name, cfg_name in candidates:
            weights_path = os.path.join(base_dir, weights_name)
            cfg_path = os.path.join(base_dir, cfg_name)
            if os.path.exists(weights_path) and os.path.exists(cfg_path):
                try:
                    net = cv2.dnn.readNet(weights_path, cfg_path)
                    # Prefer CPU by default; user can tweak later
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.net = net
                    self.mode = "yolo"
                    # try to load coco names if available
                    names_path = os.path.join(base_dir, "coco.names")
                    if os.path.exists(names_path):
                        with open(names_path, "r", encoding="utf-8") as f:
                            self.labels = [l.strip() for l in f.readlines() if l.strip()]
                    found = True
                    break
                except Exception:
                    # ignore and continue to fallback
                    self.net = None

        if not found:
            # Do NOT use `face.xml` as a fallback (per project requirement).
            # If no YOLO weights+cfg are present, leave detector uninitialized.
            self.net = None
            self.cascade = None
            self.mode = "none"

    def detect(self, frame: np.ndarray) -> Tuple[List[int], List[float], List[Tuple[int, int, int, int]]]:
        """Detect objects in `frame` and return (class_ids, scores, boxes).

        - class_ids: list[int]
        - scores: list[float]
        - boxes: list[(x, y, w, h)]
        """
        if frame is None:
            return [], [], []

        if self.mode == "yolo" and self.net is not None:
            return self._detect_yolo(frame)

        # No detector available: return empty results but log a warning
        # The tutorial expects YOLO v4 pre-trained model. To enable YOLO, place
        # the following files in the same folder as this module (or pass model_dir):
        #   - yolov4.weights
        #   - yolov4.cfg
        #   - coco.names (optional, for class labels)
        # Alternatively run: `python object_detection.py --download yolov4` to download them.
        if self.mode == "none":
            # avoid noisy logging inside loops; use a gentle message once
            if not hasattr(self, "_warned_no_model"):
                print("[object_detection] No YOLO model found. Run `python object_detection.py --download yolov4` to fetch the pre-trained weights and cfg.")
                self._warned_no_model = True
        return [], [], []

    def _detect_cascade(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detectMultiScale may return either rects or (rects, weights)
        try:
            rects = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            # older OpenCV returns just rects
            boxes = []
            scores = []
            class_ids = []
            for (x, y, w, h) in rects:
                boxes.append((int(x), int(y), int(w), int(h)))
                scores.append(1.0)
                class_ids.append(0)
            return class_ids, scores, boxes
        except Exception:
            # Try detectMultiScale2 for OpenCV versions that return weights
            try:
                rects, weights = self.cascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                boxes = []
                scores = []
                class_ids = []
                for (x, y, w, h), wscore in zip(rects, weights):
                    boxes.append((int(x), int(y), int(w), int(h)))
                    scores.append(float(wscore))
                    class_ids.append(0)
                return class_ids, scores, boxes
            except Exception:
                return [], [], []

    def _detect_yolo(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        # get output layer names in a way that's compatible across OpenCV versions
        try:
            if hasattr(self.net, "getUnconnectedOutLayersNames"):
                ln = self.net.getUnconnectedOutLayersNames()
                layer_outputs = self.net.forward(ln)
            else:
                outs = self.net.getUnconnectedOutLayers()
                layer_names = self.net.getLayerNames()
                # outs might be: array([[x], [y], ...]) or array([x, y, ...]) or list of lists
                try:
                    if isinstance(outs, np.ndarray):
                        ids = outs.flatten().astype(int).tolist()
                    else:
                        ids = []
                        for o in outs:
                            try:
                                ids.append(int(o))
                            except Exception:
                                try:
                                    ids.append(int(o[0]))
                                except Exception:
                                    pass
                except Exception:
                    ids = []

                ln = [layer_names[i - 1] for i in ids]
                layer_outputs = self.net.forward(ln)
        except Exception:
            # final fallback: try forward with no explicit layer names
            try:
                layer_outputs = self.net.forward()
            except Exception:
                # give up gracefully
                return [], [], []

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                if scores.size == 0:
                    continue
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence > self.conf_threshold:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # apply non-maxima suppression
        idxs = []
        if len(boxes) > 0:
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        final_boxes = []
        final_scores = []
        final_class_ids = []
        if len(idxs) > 0:
            # idxs can be a list of lists or a flat list depending on OpenCV
            for i in np.array(idxs).reshape(-1):
                final_boxes.append(tuple(boxes[i]))
                final_scores.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_class_ids, final_scores, final_boxes


def _download_file(url: str, dest: Path):
    """Download a file with a simple progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rDownloading {dest.name}: {percent:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _hook)
    sys.stdout.write("\n")


def download_yolo(model_dir: str | None = None, model: str = "yolov4") -> None:
    """Download pre-trained YOLO files for the tutorial.

    Supported model names: 'yolov4', 'yolov4-tiny', 'yolov3', 'yolov3-tiny'

    Note: weight files are large (~200MB). This helper only downloads files
    when requested; it does not change the repo automatically.
    """
    base = Path(model_dir) if model_dir else Path(__file__).parent
    base = base.resolve()

    urls = {
        "yolov4": {
            "weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
            "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
        },
        "yolov4-tiny": {
            "weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
            "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        },
        "yolov3": {
            "weights": "https://pjreddie.com/media/files/yolov3.weights",
            "cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        },
        "yolov3-tiny": {
            "weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
            "cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        },
    }

    names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

    if model not in urls:
        raise ValueError(f"Unknown model {model}. Available: {list(urls.keys())}")

    mapping = urls[model]
    weights_url = mapping["weights"]
    cfg_url = mapping["cfg"]

    weights_dest = base / Path(weights_url).name
    cfg_dest = base / Path(cfg_url).name
    names_dest = base / "coco.names"

    print(f"Downloading YOLO model '{model}' into {base}")
    if not weights_dest.exists():
        _download_file(weights_url, weights_dest)
    else:
        print(f"{weights_dest.name} already exists, skipping")

    if not cfg_dest.exists():
        _download_file(cfg_url, cfg_dest)
    else:
        print(f"{cfg_dest.name} already exists, skipping")

    if not names_dest.exists():
        _download_file(names_url, names_dest)
    else:
        print(f"{names_dest.name} already exists, skipping")

    print("Download finished. You can now create ObjectDetection() and it will load YOLO files if they are in the module folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="object_detection helper")
    parser.add_argument("--download", nargs="?", const="yolov4", help="Download YOLO model (name optional, default yolov4)")
    parser.add_argument("--model-dir", help="Directory to store model files (defaults to module folder)")
    args = parser.parse_args()
    if args.download:
        download_yolo(args.model_dir, args.download)


__all__ = ["ObjectDetection"]
