"""Summarise a clear-the-centre sweep: detection, planning, execution, collateral movement.

    .venv/bin/python summarize_sidesort_sweep.py --dir runs/sidesort_sweep
Ground truth is used here for scoring only, after the runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def move_mm(report, name):
    return float(np.linalg.norm(np.array(report["after"][name]["centre"]) - report["before"][name]["centre"]) * 1000)


def summarise(run):
    row = {"run": run.name}
    gt, ev, plan = (load(run / f) for f in ("ground_truth.json", "perception_evaluation.json", "sort_plan.json"))
    report, events = load(run / "execution_report.json"), load(run / "execution_events.json") or []
    if gt is None:
        return dict(row, stage="capture failed")
    count = lambda k: sum(o["category"] == k for o in gt["objects"])
    row["wide_caps"] = sum(o.get("orientation") == "wide_end_up" for o in gt["objects"])
    if ev is None or plan is None:
        return dict(row, stage="detection/planning failed")
    found = lambda k: f'{sum(m["category"] == k for m in ev["matches"])}/{count(k)}'
    row.update(cubes_found=found("cube"), vials_found=found("vial"), caps_found=found("cap"),
               cap_orient_ok=f'{ev["cap_orientations_correct"]}/{ev["caps_matched"]}',
               false_detections=len(ev["unmatched_detections"]))
    planned = {k: sum(p["category"] == k for p in plan["picks"]) for k in ("vial", "cube", "cap")}
    row["planned"] = planned
    row["skipped"] = sorted({f'{s["category"]}:{s["skip_reason"]}' for s in plan["skipped"] if s.get("skip_reason")})
    if report is None:
        return dict(row, stage="execution failed (no report)")
    row.update(stage="executed", task_complete=bool(report["task_complete"]), abort=report["abort_reason"],
               pickups=f'{report["verified_pickups"]}/{len(plan["picks"])}',
               placements=f'{report["verified_placements"]}/{len(plan["picks"])}')
    place = {p["detection_id"]: p["place_xy"] for p in plan["picks"]}
    errs, failures, rejects = {"vial": [], "cube": []}, [], 0
    for e in events:
        name, cat = e.get("tracked_object"), e["category"]
        if name in report["after"] and e["detection_id"] in place and cat in errs:
            fin = np.array(report["after"][name]["centre"][:2])
            errs[cat].append(round(float(np.linalg.norm(fin - place[e["detection_id"]]) * 1000), 1))
        if not e["pickup_verified"]:
            failures.append(f"{name} pickup")
        elif not e["placement_verified"]:
            failures.append(f"{name} placement")
        rejects += bool(e.get("wrist_abort"))
    row.update(vial_err_mm=errs["vial"], cube_err_mm=errs["cube"], failures=failures, wrist_rejects=rejects)
    caps = [n for n in report["before"] if n.startswith("SortCap")]
    row["caps_max_move_mm"] = round(max((move_mm(report, n) for n in caps), default=0.0), 2)
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
        print(
            f'{r["run"]}: {"COMPLETE" if r["task_complete"] else "INCOMPLETE"}  found cubes {r["cubes_found"]} '
            f'vials {r["vials_found"]} caps {r["caps_found"]} (orient {r["cap_orient_ok"]}, fp {r["false_detections"]}) | '
            f'planned {r["planned"]} | pick {r["pickups"]} place {r["placements"]} | '
            f'cube err {r["cube_err_mm"]} vial err {r["vial_err_mm"]} | caps moved {r["caps_max_move_mm"]} mm'
            + (f' | FAILED: {r["failures"]}' if r["failures"] else "")
            + (f' | wrist rejects {r["wrist_rejects"]}' if r["wrist_rejects"] else "")
            + (f' | abort: {r["abort"]}' if r["abort"] else "")
        )
    if done:
        cube = [e for r in done for e in r["cube_err_mm"]]
        vial = [e for r in done for e in r["vial_err_mm"]]
        print(
            f"task complete: {sum(r['task_complete'] for r in done)}/{len(done)} layouts; "
            f"caps untouched (<2 mm): {sum(r['caps_max_move_mm'] < 2 for r in done)}/{len(done)}"
        )
        if cube:
            print(f"cube slot error: n={len(cube)} mean {np.mean(cube):.1f} max {np.max(cube):.1f} mm")
        if vial:
            print(f"vial slot error: n={len(vial)} mean {np.mean(vial):.1f} max {np.max(vial):.1f} mm")


if __name__ == "__main__":
    main()
