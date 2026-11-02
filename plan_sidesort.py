"""Plan the clear-the-centre task: tubes to the -y side, cubes to the +y side, caps stay.

A wrapper over plan_sort (unchanged), run twice over the same detections:
  * sort mode for the tubes (caps are also assessed there and left in place);
  * cube_stack mode only for its per-cube measurement and gates (position, top-face
    yaw, pose confidence, silhouette fit). Each cube's own verdict is kept; the tower
    logic is ignored and every usable cube gets a side slot of its own.

    .venv/bin/python plan_sidesort.py --run runs/example
"""

import argparse
import json
from pathlib import Path

import cv2

import capvial_config as K
import plan_capvial
import plan_sort
import scene_config as C
from backproject import CameraModel

# cube_stack mode marks every cube with this when the colours are not exactly 2/2/2;
# it says nothing about the individual cube, so it does not block a side placement.
NOT_A_TOWER = "incomplete_cube_colour_set"


def usable_cube(o):
    return o.get("category") == "cube" and "yaw_deg" in o and o.get("skip_reason") in (None, NOT_A_TOWER)


def cube_picks(observations, slots):
    """One pick per usable cube: nearest the +y side first, slots in order."""
    cubes = sorted((o for o in observations if usable_cube(o)), key=lambda o: -o["xy"][1])
    picks, skipped = [], []
    for i, o in enumerate(cubes):
        if i >= len(slots):
            skipped.append(dict(o, eligible=False, skip_reason="no_cube_destination"))
            continue
        picks.append(
            dict(
                o,
                eligible=True,
                skip_reason=None,
                side_slot=i,
                place_xy=list(slots[i]),
                place_z=C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE,
                place_spread_dir=[0.0, 1.0, 0.0],  # tidy: every cube set down axis-aligned
                place_yaw_deg=0.0,
            )
        )
    return picks, skipped


def build(det, cam, img_raw, img_norm):
    for name, value in plan_capvial.SLOT_OVERRIDES.items():
        setattr(C, name, value)
    sorted_plan = plan_sort.build_plan(det, cam, img_raw, task="sort")
    cube_plan = plan_sort.build_plan(det, cam, img_norm, task="cube_stack")
    picks, over = cube_picks(cube_plan["observations"], K.CUBE_SLOTS)
    not_cube = lambda o: o.get("category") != "cube"
    unusable = [o for o in cube_plan["skipped"] if o.get("category") == "cube" and not usable_cube(o)]
    plan = dict(sorted_plan)
    plan["task"] = "sort"
    plan["picks"] = [p for p in sorted_plan["picks"] if not_cube(p)] + picks
    plan["observations"] = [o for o in sorted_plan["observations"] if not_cube(o)] + [
        o for o in cube_plan["observations"] if o.get("category") == "cube"
    ]
    plan["skipped"] = [o for o in sorted_plan["skipped"] if not_cube(o)] + unusable + over
    plan["cube_slots"] = [list(s) for s in K.CUBE_SLOTS]
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    cam = CameraModel(str(run / "camera.json"))
    raw = cv2.imread(str(run / "world_camera.png"))
    norm = cv2.imread(str(run / "world_camera_norm.png"))
    if raw is None or norm is None:
        raise SystemExit(f"Need world_camera.png and world_camera_norm.png in {run} (run perceive_opencv.py)")
    det = json.loads((run / "detections.json").read_text())
    plan = build(det, cam, raw, norm)
    (run / "sort_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    gt_path = run / "ground_truth.json"
    if gt_path.exists():  # scored only after the plan is saved
        evaluation = plan_sort.score_observations(plan, json.loads(gt_path.read_text()))
        (run / "perception_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    kinds = {k: sum(p["category"] == k for p in plan["picks"]) for k in ("vial", "cube", "cap")}
    print(f"[sidesort plan] picks {kinds}; skipped {sorted({(s['category'], s['skip_reason']) for s in plan['skipped']})}")


if __name__ == "__main__":
    main()
