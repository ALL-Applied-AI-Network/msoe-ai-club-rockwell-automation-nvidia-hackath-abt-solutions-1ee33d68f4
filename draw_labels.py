"""Draw YOLO-OBB labels on dataset images and tile them into one contact sheet.

    .venv/bin/python draw_labels.py --dataset datasets/capvial --split train --out sheet.png [--n 8] [--cols 2]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["cap_narrow_up", "cap_wide_up", "vial"]
COLOURS = [(0, 255, 0), (255, 0, 255), (0, 200, 255)]  # BGR per class


def draw(image, label_file):
    h, w = image.shape[:2]
    if not label_file.exists():
        return image
    for line in label_file.read_text().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        pts = (np.array(parts[1:], float).reshape(4, 2) * [w, h]).astype(np.int32)
        cv2.polylines(image, [pts], True, COLOURS[cls], 2, cv2.LINE_AA)
        cv2.putText(image, CLASSES[cls], tuple(pts[0] + [0, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[cls], 2, cv2.LINE_AA)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--width", type=int, default=960, help="width of each tile")
    a = ap.parse_args()
    root = Path(a.dataset)
    files = sorted((root / "images" / a.split).glob("*.png"))[: a.n]
    if not files:
        raise SystemExit(f"No images under {root / 'images' / a.split}")
    tiles = []
    for f in files:
        im = draw(cv2.imread(str(f)), root / "labels" / a.split / (f.stem + ".txt"))
        tiles.append(cv2.resize(im, (a.width, round(im.shape[0] * a.width / im.shape[1]))))
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + a.cols]) for i in range(0, len(tiles), a.cols)]
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"wrote {a.out} ({len(files)} images)")


if __name__ == "__main__":
    main()
