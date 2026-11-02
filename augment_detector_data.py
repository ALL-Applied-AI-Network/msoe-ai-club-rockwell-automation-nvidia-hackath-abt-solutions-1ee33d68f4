"""Build a corrupted copy of the cap/vial training set (dark, dim, JPEG, noisy, recoloured table).

Each training image gets 1-3 random corruptions with randomised strength; labels are
unchanged because the corruptions do not move pixels. The validation split stays clean.
The evaluation CASES in perturb_eval.py are fixed points; the ranges here cover them
without being identical to them for most draws.

    .venv/bin/python augment_detector_data.py --src datasets/capvial --out datasets/capvial_aug [--seed 7]
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

import perturb_eval as P


def random_corruptions(rng):
    """A list of image functions; table recolouring comes first so later steps act on it."""
    fns = []
    if rng.random() < 0.35:
        # Mostly neutral/dark tints, sometimes coloured.
        base = rng.uniform(0.3, 1.0)
        tint = np.clip(base * rng.uniform(0.85, 1.15, 3), 0.2, 1.0) * 255
        fns.append(P.table_tint(tuple(tint)))
    if rng.random() < 0.5:
        fns.append(P.gain(float(np.exp(rng.uniform(np.log(0.25), np.log(1.5))))))
    if rng.random() < 0.2:
        fns.append(P.shadow_gradient(float(rng.uniform(0.2, 0.6))))
    if rng.random() < 0.2:
        fns.append(P.blur(int(rng.choice([3, 5, 7]))))
    if rng.random() < 0.5:
        fns.append(P.noise(float(rng.uniform(3, 30))))
    if rng.random() < 0.55:
        fns.append(P.jpeg(int(rng.integers(10, 60))))
    return fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean-frac", type=float, default=0.15, help="fraction left unmodified")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    rng = np.random.default_rng(a.seed)
    for sub in ("labels", "meta"):
        shutil.copytree(src / sub, out / sub)
    shutil.copytree(src / "images" / "val", out / "images" / "val")
    (out / "images" / "train").mkdir(parents=True)
    count = 0
    for f in sorted((src / "images" / "train").glob("*.png")):
        im = cv2.imread(str(f))
        if rng.random() >= a.clean_frac:
            for fn in random_corruptions(rng):
                im = fn(im, rng)
            count += 1
        cv2.imwrite(str(out / "images" / "train" / f.name), im)
    print(f"{count} train images corrupted")


if __name__ == "__main__":
    main()
