"""Random, valid layouts for the cap/vial task.

Same object counts as the default scene (6 cubes, 2 vials, 2 caps) with random
positions, tube angles, cube angles and cap orientations. Spacing follows the
default cube layout's rules (scene_config.hackathon_cubes): enough room that the
fingers never touch a neighbour. Tubes and caps stay clear of the set-down slots.

    .venv/bin/python capvial_layouts.py --seeds 1 2 3 --out-dir layouts/
"""

import argparse
import json
from pathlib import Path

import numpy as np

import capvial_config as K
import scene_config as C

CUBE_COLOURS = ("yellow", "yellow", "red", "red", "blue", "blue")  # default scene order
CUBE_X, CUBE_Y = K.CUBE_FIELD_X, K.CUBE_FIELD_Y
VIAL_X, VIAL_Y, VIAL_YAW = (0.30, 0.46), (-0.18, 0.20), (-70.0, 70.0)
CAP_X, CAP_Y = (0.24, 0.46), (-0.15, 0.25)
N_CUBES, N_VIALS, N_CAPS = 6, 2, 2

# Minimum clear distance between object footprints (metres, axis/centre to centre).
MIN_GAP = {
    ("cube", "cube"): 0.04,
    ("cube", "cap"): 0.065,
    ("cube", "vial"): 0.055,
    ("vial", "vial"): 0.06,
    ("vial", "cap"): 0.06,
    ("cap", "cap"): 0.05,
}


def footprint(kind, x, y, yaw=0.0):
    """Points sampling an object's table footprint (a tube is a line segment)."""
    if kind == "vial":
        axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        t = np.linspace(-C.VIAL_LENGTH / 2, C.VIAL_LENGTH / 2, 29)
        return np.array([x, y]) + t[:, None] * axis
    return np.array([[x, y]])


def gap(a, b):
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min())


def min_gap(kind_a, kind_b):
    return MIN_GAP.get((kind_a, kind_b)) or MIN_GAP[(kind_b, kind_a)]


def sample(seed, p_wide_up=0.5):
    rng = np.random.default_rng(seed)
    placed = []  # (kind, footprint)

    def place(kind, xr, yr, yaw_range=None):
        for _ in range(20000):
            x, y = rng.uniform(*xr), rng.uniform(*yr)
            yaw = rng.uniform(*yaw_range) if yaw_range else 0.0
            fp = footprint(kind, x, y, yaw)
            if all(gap(fp, other) >= min_gap(kind, k) for k, other in placed):
                placed.append((kind, fp))
                return x, y, yaw
        raise ValueError(f"could not place a {kind} for seed {seed}")

    vials = [place("vial", VIAL_X, VIAL_Y, VIAL_YAW) for _ in range(N_VIALS)]
    caps = [place("cap", CAP_X, CAP_Y)[:2] for _ in range(N_CAPS)]
    cubes = [place("cube", CUBE_X, CUBE_Y)[:2] for _ in range(N_CUBES)]
    return {
        "seed": int(seed),
        "cubes": [[round(x, 6), round(y, 6), colour] for (x, y), colour in zip(cubes, CUBE_COLOURS)],
        "cube_yaws_deg": [round(float(v), 3) for v in rng.uniform(-45, 45, N_CUBES)],
        "vials": [[round(x, 6), round(y, 6), round(yaw, 3)] for x, y, yaw in vials],
        "caps": [[round(x, 6), round(y, 6)] for x, y in caps],
        "cap_orientations": [
            "wide_end_up" if rng.random() < p_wide_up else "narrow_end_up" for _ in range(N_CAPS)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--p-wide-up", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        (out / f"layout_{seed}.json").write_text(json.dumps(sample(seed, a.p_wide_up), indent=2) + "\n")
    print(f"wrote {len(a.seeds)} layouts to {out}")


if __name__ == "__main__":
    main()
