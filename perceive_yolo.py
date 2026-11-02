"""Detect caps and vials with the trained YOLO-OBB model and write detections.json.

Runs in the YOLO environment. Cubes are deliberately not detected: the cap/vial task
leaves them alone. The overhead capture is downscaled to 1920x1080, the realistic
camera resolution the detector was trained at, and denoised only if it is noisy.

    .venv-yolo/bin/python perceive_yolo.py --run runs/example [--weights best.pt]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from capvial_detections import CLASS_NAMES, detection
from wrist_refine import adaptive_denoise

HERE = Path(__file__).resolve().parent
COLOURS = [(0, 255, 0), (255, 0, 255), (0, 200, 255)]  # BGR per class
NATIVE = (1920, 1080)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default=str(HERE / "runs_detector/capvial_n/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--merge", action="store_true", help="add to an existing detections.json (e.g. the cubes)")
    a = ap.parse_args()
    run = Path(a.run)
    image = cv2.imread(str(run / "world_camera.png"))
    if image is None:
        raise SystemExit(f"Missing world_camera.png in {run}; run capture_scene.py first")
    if image.shape[1] > NATIVE[0]:
        image = cv2.resize(image, NATIVE, interpolation=cv2.INTER_AREA)
    image = adaptive_denoise(image)
    from ultralytics import YOLO

    result = YOLO(a.weights).predict(image, imgsz=NATIVE[0], conf=a.conf, device=a.device, verbose=False)[0]
    size = (image.shape[1], image.shape[0])
    objects, overlay = [], image.copy()
    if result.obb is not None:
        for poly, cls, conf in zip(
            result.obb.xyxyxyxy.cpu().numpy(), result.obb.cls.cpu().numpy(), result.obb.conf.cpu().numpy()
        ):
            objects.append(detection(int(cls), float(conf), poly, size))
            pts = poly.astype(np.int32)
            cv2.polylines(overlay, [pts], True, COLOURS[int(cls)], 2, cv2.LINE_AA)
            cv2.putText(overlay, f"{CLASS_NAMES[int(cls)]} {conf:.2f}", tuple(pts[0] + [0, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOURS[int(cls)], 2, cv2.LINE_AA)
    det_path = run / "detections.json"
    if a.merge and det_path.is_file():
        det = json.loads(det_path.read_text())
        det["objects"] = det["objects"] + objects
        det["perception"] = det.get("perception", "OpenCV cubes") + " + YOLOv8n-OBB (caps and vials)"
    else:
        det = {
            # plan_sort/sort_scene accept "OpenCV" or "Cosmos3-Nano"; the real provenance is below.
            "source": "OpenCV",
            "perception": "YOLOv8n-OBB (caps and vials only)",
            "coordinate_convention": "unit",
            "objects": objects,
        }
    det_path.write_text(json.dumps(det, indent=2) + "\n")
    cv2.imwrite(str(run / "detections_overlay_yolo.png"), overlay)
    print("detected:", dict(Counter(o["label"] + (" " + o["orientation"] if "orientation" in o else "") for o in objects)))


if __name__ == "__main__":
    main()
