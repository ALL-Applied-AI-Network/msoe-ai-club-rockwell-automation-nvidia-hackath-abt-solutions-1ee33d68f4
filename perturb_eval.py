"""Stress-test the OpenCV cube detector with synthetic image corruptions.

Perturbs a captured table image (sensor noise, lighting, blur, compression,
recoloured table), reruns detection + planning, then scores against ground truth
for EVALUATION only. Host interpreter; no Isaac needed.

    .venv/bin/python perturb_eval.py --run runs/demo_01 [--save-dir runs/demo_01/perturbed]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import scene_config as C
from backproject import CameraModel
from perceive_opencv import find_cubes, normalise
from plan_sort import build_plan, score_observations


def noise(sigma):
    def f(im, rng):
        out = im.astype(np.float32) + rng.normal(0, sigma, im.shape)
        return np.clip(out, 0, 255).astype(np.uint8)

    return f


def gain(g):
    return lambda im, rng: np.clip(im.astype(np.float32) * g, 0, 255).astype(np.uint8)


def blur(k):
    return lambda im, rng: cv2.GaussianBlur(im, (k, k), 0)


def jpeg(q):
    def f(im, rng):
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return f


def table_tint(bgr):
    """Recolour the pale table/backdrop (low saturation, bright) toward `bgr`."""

    def f(im, rng):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        pale = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 120)
        out = im.astype(np.float32)
        tint = np.array(bgr, np.float32) / 255.0
        out[pale] = out[pale] * tint
        return np.clip(out, 0, 255).astype(np.uint8)

    return f


def shadow_gradient(strength):
    def f(im, rng):
        H, W = im.shape[:2]
        ramp = np.linspace(1 - strength, 1.0, W, dtype=np.float32)[None, :, None]
        return np.clip(im.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

    return f


def stacked(*fs):
    def f(im, rng):
        for g in fs:
            im = g(im, rng)
        return im

    return f


CASES = {
    "baseline": lambda im, rng: im,
    "noise_s5": noise(5),
    "noise_s15": noise(15),
    "noise_s30": noise(30),
    "dim_0.5": gain(0.5),
    "dim_0.3": gain(0.3),
    "bright_1.4": gain(1.4),
    "blur_5": blur(5),
    "blur_9": blur(9),
    "jpeg_q20": jpeg(20),
    "table_beige": table_tint((190, 215, 235)),
    "table_light_blue": table_tint((255, 225, 190)),
    "table_green": table_tint((190, 240, 190)),
    "table_dark_grey": table_tint((90, 90, 90)),
    "table_yellowish": table_tint((140, 230, 250)),
    "table_reddish": table_tint((170, 170, 245)),
    "shadow_gradient_0.5": shadow_gradient(0.5),
    "noise15+dim0.5+jpeg": stacked(noise(15), gain(0.5), jpeg(30)),
}


def run_case(name, fn, im0, cam, gt, rng, save_dir, use_norm):
    im = fn(im0, rng)
    if save_dir:
        cv2.imwrite(str(save_dir / f"{name}.png"), im)
    if use_norm:
        im = normalise(im, cam)  # detector and planner both see the corrected image
    objects, rejected = find_cubes(im, cam)
    det = dict(source="OpenCV", coordinate_convention="unit", objects=objects)
    try:
        plan = build_plan(det, cam, im, task="cube_stack")
        ev = score_observations(plan, gt)  # evaluation only, after planning
    except Exception as e:  # report, never hide
        return dict(case=name, error=f"{type(e).__name__}: {e}")
    picks = plan["picks"]
    colours = {c: sum(o["colour"] == c for o in objects) for c in ("yellow", "blue", "red")}
    matched = [m for m in ev["matches"] if m.get("selected")]
    return dict(
        case=name,
        blobs=len(objects),
        colours=colours,
        picks=len(picks),
        stack_ready=bool(plan["stack_ready"]),
        max_pos_err_mm=round(max((m["position_error_mm"] for m in matched), default=float("nan")), 2),
        max_yaw_err_deg=round(max((m["yaw_error_deg_mod90"] for m in matched), default=float("nan")), 2),
        colour_wrong=sum(not m["colour_correct"] for m in matched),
        skipped=sorted({s["skip_reason"] for s in plan["skipped"]}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalise", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    im0 = cv2.imread(str(run / "world_camera.png"))
    cam = CameraModel(str(run / "camera.json"))
    gt = json.loads((run / "ground_truth.json").read_text())
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = [run_case(n, f, im0, cam, gt, rng, save, a.normalise) for n, f in CASES.items()]
    print(f"{'case':24} picks ready  pos_mm yaw_deg  colour_err  note")
    for r in rows:
        if "error" in r:
            print(f"{r['case']:24} ERROR {r['error']}")
            continue
        ok = "PASS" if r["stack_ready"] and r["picks"] == 6 and r["colour_wrong"] == 0 else "FAIL"
        print(
            f"{r['case']:24} {r['picks']:>3}  {str(r['stack_ready']):5} {r['max_pos_err_mm']:>7} "
            f"{r['max_yaw_err_deg']:>7} {r['colour_wrong']:>6}      {ok} {r['colours']} {r['skipped'] or ''}"
        )
    (run / ("perturbation_results_norm.json" if a.normalise else "perturbation_results.json")).write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
