"""Summarise a cap/vial sweep: detection, planning, execution and collateral movement.

    .venv/bin/python summarize_capvial_sweep.py --dir runs/capvial_sweep
Ground truth is used here for scoring only, after the runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def displacement_mm(report, names):
    return {
        n: float(np.linalg.norm(np.array(report["after"][n]["centre"]) - report["before"][n]["centre"]) * 1000)
        for n in names
    }


def summarise(run):
    row = {"run": run.name}
    gt = load(run / "ground_truth.json")
    ev = load(run / "perception_evaluation.json")
    plan = load(run / "sort_plan.json")
    report = load(run / "execution_report.json")
    events = load(run / "execution_events.json") or []
    if gt is None:
        return dict(row, stage="capture failed")
    n_vial = sum(o["category"] == "vial" for o in gt["objects"])
    n_cap = sum(o["category"] == "cap" for o in gt["objects"])
    row["wide_caps"] = sum(o.get("orientation") == "wide_end_up" for o in gt["objects"])
    if ev is None or plan is None:
        return dict(row, stage="detection/planning failed")
    matched = [m for m in ev["matches"] if m["category"] in ("cap", "vial")]
    row["vials_found"] = f'{sum(m["category"] == "vial" for m in matched)}/{n_vial}'
    row["caps_found"] = f'{sum(m["category"] == "cap" for m in matched)}/{n_cap}'
    row["cap_orient_ok"] = f'{ev["cap_orientations_correct"]}/{ev["caps_matched"]}'
    row["false_detections"] = len(ev["unmatched_detections"])
    row["det_err_mm_max"] = round(max((m["position_error_mm"] for m in matched), default=float("nan")), 1)
    row["planned_vials"] = sum(p["category"] == "vial" for p in plan["picks"])
    row["planned_caps"] = sum(p["category"] == "cap" for p in plan["picks"])
    row["skipped"] = sorted({s["skip_reason"] for s in plan["skipped"] if s.get("skip_reason")})
    if report is None:
        return dict(row, stage="execution failed (no report)")
    row["stage"] = "executed"
    row["task_complete"] = bool(report["task_complete"])
    row["pickups"] = f'{report["verified_pickups"]}/{len(plan["picks"])}'
    row["placements"] = f'{report["verified_placements"]}/{len(plan["picks"])}'
    row["abort"] = report["abort_reason"]
    place = {p["detection_id"]: p["place_xy"] for p in plan["picks"]}
    errs = []
    for e in events:
        if e.get("tracked_object") in report["after"] and e["detection_id"] in place:
            fin = np.array(report["after"][e["tracked_object"]]["centre"][:2])
            errs.append(round(float(np.linalg.norm(fin - place[e["detection_id"]]) * 1000), 1))
    row["slot_err_mm"] = errs
    cubes = displacement_mm(report, [n for n in report["before"] if n.startswith("SortCube")])
    row["cube_max_move_mm"] = round(max(cubes.values(), default=0.0), 2)
    picked = {e.get("tracked_object") for e in events}
    others = displacement_mm(report, [n for n in report["before"] if n.startswith(("SortCap", "SortVial")) and n not in picked])
    row["bystander_max_move_mm"] = round(max(others.values(), default=0.0), 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    root = Path(ap.parse_args().dir)
    rows = [summarise(p) for p in sorted(root.glob("seed_*")) if p.is_dir()]
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    done = [r for r in rows if r.get("stage") == "executed"]
    print(f"{len(rows)} layouts, {len(done)} executed")
    for r in rows:
        if r.get("stage") != "executed":
            print(f'{r["run"]}: {r.get("stage")}')
            continue
        flag = "COMPLETE" if r["task_complete"] else "INCOMPLETE"
        print(
            f'{r["run"]}: {flag}  vials {r["vials_found"]} caps {r["caps_found"]} orient {r["cap_orient_ok"]} '
            f'fp {r["false_detections"]} | planned v{r["planned_vials"]}/c{r["planned_caps"]} '
            f'pick {r["pickups"]} place {r["placements"]} slot_err {r["slot_err_mm"]} | '
            f'cubes moved {r["cube_max_move_mm"]} mm, bystanders {r["bystander_max_move_mm"]} mm'
            + (f' | abort: {r["abort"]}' if r["abort"] else "")
        )
    if done:
        print(
            f"task complete: {sum(r['task_complete'] for r in done)}/{len(done)}; "
            f"cubes untouched (<2 mm) in {sum(r['cube_max_move_mm'] < 2 for r in done)}/{len(done)}; "
            f"wide-up caps seen in {sum(r['wide_caps'] > 0 for r in rows if 'wide_caps' in r)} layouts"
        )


if __name__ == "__main__":
    main()
