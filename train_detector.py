"""Train the cap/vial oriented-box detector on the synthetic dataset.

Runs in the YOLO environment, not the host or Isaac interpreters:

    .venv-yolo/bin/python train_detector.py --dataset datasets/capvial --name capvial_n \
        [--model yolov8n-obb.pt] [--epochs 50] [--device 0]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

NAMES = ["cap_narrow_up", "cap_wide_up", "vial"]


def write_data_yaml(dataset):
    root = Path(dataset).resolve()
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    path = root / "data.yaml"
    path.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolov8n-obb.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1920, help="native render width; objects are small")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs_detector")
    a = ap.parse_args()
    data = write_data_yaml(a.dataset)
    model = YOLO(a.model)
    model.train(
        data=str(data),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        device=a.device,
        project=str(Path(a.project).resolve()),
        name=a.name,
        # Rotating or vertically flipping would change the calibrated camera's geometry.
        degrees=0.0,
        flipud=0.0,
        scale=0.25,
        patience=15,
        workers=4,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
