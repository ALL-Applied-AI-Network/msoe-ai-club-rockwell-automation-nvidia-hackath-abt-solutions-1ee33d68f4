"""Score the cap/vial detector against ground truth, clean and under image corruptions.

Metrics are the ones the planner needs, not box overlap: per-group recall and false
positives, cap orientation accuracy, cap position error (mm) and vial axis angle and
endpoint error (mm). Ground truth comes from the dataset labels and meta files.

    .venv-yolo/bin/python eval_detector.py --weights runs_detector/capvial_n/weights/best.pt \
        --dataset datasets/capvial --split val --camera runs/demo_01/camera.json \
        [--cases baseline noise_s15 ...] [--n 100]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import scene_config as C
from backproject import CameraModel
from perturb_eval import CASES
from wrist_refine import adaptive_denoise

CAP_CLASSES = (0, 1)
VIAL_CLASS = 2
Z = C.TABLE_TOP_Z + C.SURFACE_DZ


def group(cls):
    return "vial" if cls == VIAL_CLASS else "cap"


def iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def load_gt(label_file, meta_file, w, h):
    meta = json.loads(meta_file.read_text())
    vials = [m for m in meta["objects"] if m["cls"] == "vial"]
    gt = []
    for line in label_file.read_text().split("\n"):
        p = line.split()
        if not p:
            continue
        poly = np.array(p[1:], float).reshape(4, 2) * [w, h]
        gt.append(dict(cls=int(p[0]), poly=poly, ends=None))
    for g in gt:  # attach true axis endpoints to the vial whose midpoint is nearest
        if g["cls"] != VIAL_CLASS or not vials:
            continue
        centre = g["poly"].mean(0)
        best = min(vials, key=lambda m: np.linalg.norm(np.mean(m["axis_endpoints_px"], 0) - centre))
        g["ends"] = np.array(best["axis_endpoints_px"])
    return gt


def vial_endpoints(poly):
    """Midpoints of the two short sides of an oriented box (its axis ends)."""
    edges = [(poly[i], poly[(i + 1) % 4]) for i in range(4)]
    lengths = [np.linalg.norm(b - a) for a, b in edges]
    short = sorted(range(4), key=lambda i: lengths[i])[:2]
    return np.array([(edges[i][0] + edges[i][1]) / 2 for i in short])


def to_world(cam, px, z):
    p = cam.to_plane(px[0], px[1], z=z)
    return None if p is None else np.asarray(p[:2])


def evaluate(results, gts, cam):
    stats = dict(cap_gt=0, cap_hit=0, cap_orient_ok=0, vial_gt=0, vial_hit=0, fp=0, images=0)
    cap_err, vial_end_err, vial_ang_err = [], [], []
    for res, gt in zip(results, gts):
        stats["images"] += 1
        preds = []
        if res.obb is not None and len(res.obb):
            for poly, cls in zip(res.obb.xyxyxyxy.cpu().numpy(), res.obb.cls.cpu().numpy()):
                preds.append(dict(cls=int(cls), poly=poly))
        used = set()
        for g in gt:
            grp = group(g["cls"])
            stats[f"{grp}_gt"] += 1
            options = [(iou(g["poly"], p["poly"]), i) for i, p in enumerate(preds) if i not in used and group(p["cls"]) == grp]
            score, i = max(options, default=(0.0, None))
            if score < 0.3:
                continue
            used.add(i)
            p = preds[i]
            stats[f"{grp}_hit"] += 1
            if grp == "cap":
                stats["cap_orient_ok"] += p["cls"] == g["cls"]
                a = to_world(cam, g["poly"].mean(0), Z + C.CAP_HEIGHT / 2)
                b = to_world(cam, p["poly"].mean(0), Z + C.CAP_HEIGHT / 2)
                if a is not None and b is not None:
                    cap_err.append(np.linalg.norm(a - b) * 1000)
            elif g["ends"] is not None:
                pe, ge = vial_endpoints(p["poly"]), g["ends"]
                pairs = min((pe, pe[::-1]), key=lambda q: np.linalg.norm(q - ge, axis=1).sum())
                zc = Z + C.VIAL_RADIUS + 0.001
                w = [(to_world(cam, x, zc), to_world(cam, y, zc)) for x, y in zip(pairs, ge)]
                if all(a is not None and b is not None for a, b in w):
                    vial_end_err.append(np.mean([np.linalg.norm(a - b) for a, b in w]) * 1000)
                va, vb = pairs[1] - pairs[0], ge[1] - ge[0]
                ang = np.degrees(np.arccos(np.clip(abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), 0, 1)))
                vial_ang_err.append(ang)
        stats["fp"] += len(preds) - len(used)
    return stats, cap_err, vial_end_err, vial_ang_err


def fmt(stats, cap_err, vial_end_err, vial_ang_err):
    r = lambda hit, n: f"{hit / max(n, 1):5.1%}"
    m = lambda v: f"{np.mean(v):4.1f}/{np.max(v):4.1f}" if len(v) else "  --  "
    return (
        f"cap rec {r(stats['cap_hit'], stats['cap_gt'])}  orient {r(stats['cap_orient_ok'], stats['cap_hit'])}  "
        f"pos mm {m(cap_err)} | vial rec {r(stats['vial_hit'], stats['vial_gt'])}  end mm {m(vial_end_err)}  "
        f"ang deg {m(vial_ang_err)} | FP/img {stats['fp'] / max(stats['images'], 1):.2f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--camera", required=True, help="camera.json; resolution is overridden to 1920x1080")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0, help="limit images (0 = all)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--denoise", action="store_true", help="noise-adaptive denoise before inference")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    cam_json = json.loads(Path(a.camera).read_text())
    cam_json["resolution"] = [1920, 1080]  # same pinhole, 1080p render
    cam = CameraModel(cam_json)
    root = Path(a.dataset)
    files = sorted((root / "images" / a.split).glob("*.png"))
    files = files[: a.n] if a.n else files
    images = [cv2.imread(str(f)) for f in files]
    h, w = images[0].shape[:2]
    gts = [load_gt(root / "labels" / a.split / (f.stem + ".txt"), root / "meta" / a.split / (f.stem + ".json"), w, h) for f in files]
    cases = a.cases or list(CASES)
    rows = {}
    print(f"{len(files)} images, {sum(len(g) for g in gts)} labelled objects")
    for name in cases:
        rng = np.random.default_rng(0)
        pert = [CASES[name](im, rng) for im in images]
        if a.denoise:
            pert = [adaptive_denoise(im) for im in pert]
        results = []
        for i in range(0, len(pert), 8):
            results += model.predict(pert[i : i + 8], imgsz=1920, conf=a.conf, device=a.device, verbose=False)
        out = evaluate(results, gts, cam)
        rows[name] = dict(stats=out[0], cap_err_mm=out[1], vial_end_err_mm=out[2], vial_ang_deg=out[3])
        print(f"{name:22s} {fmt(*out)}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
