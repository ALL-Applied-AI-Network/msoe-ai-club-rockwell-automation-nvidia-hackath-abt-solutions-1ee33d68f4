#!/usr/bin/env bash
# Capture -> YOLO cap/vial detection -> plan -> pick caps and vials and set them aside -> videos.
# The cubes are left alone. Independent of run_opencv_pipeline.sh (the cube stack).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
YOLO_PYTHON="${YOLO_PYTHON:-$HERE/.venv-yolo/bin/python}"
for executable in "$ISAAC/python.sh" "$HOST_PYTHON" "$YOLO_PYTHON"; do
    if [ ! -x "$executable" ]; then
        echo "Python launcher missing or not executable: $executable" >&2
        exit 1
    fi
done
RUN="${1:-$HERE/runs/capvial_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN"
RUN="$(cd "$RUN" && pwd)"
if [ -n "$(ls -A "$RUN")" ]; then
    echo "Use an empty run directory: $RUN" >&2
    exit 1
fi
"$ISAAC/python.sh" "$HERE/capture_scene.py" --out "$RUN"
"$YOLO_PYTHON" "$HERE/perceive_yolo.py" --run "$RUN"
"$HOST_PYTHON" "$HERE/plan_capvial.py" --run "$RUN"
# sort_capvial.py is the cap/vial executor (a copy of sort_scene.py, which stays unchanged).
# "slide" frees a set-down tube by sliding the open gripper off its end; lifting straight
# up drags the tube along because the open fingertip gap is narrower than the tube.
"$ISAAC/python.sh" "$HERE/sort_capvial.py" --run "$RUN" --vial-release "${VIAL_RELEASE:-slide}"
# Isaac shutdown can return zero even after an exception; check the result.
"$HOST_PYTHON" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
report = run / 'execution_report.json'
if not report.is_file():
    raise SystemExit('Execution report missing; inspect the simulator output.')
result = json.loads(report.read_text())
plan = json.loads((run / 'sort_plan.json').read_text())
cubes = [n for n in result['before'] if n.startswith('SortCube')]
moved = max((np.linalg.norm(np.array(result['after'][n]['centre']) - result['before'][n]['centre']) * 1000 for n in cubes), default=0.0)
print({k: result.get(k) for k in ('task_complete', 'verified_pickups', 'verified_placements', 'abort_reason')})
print('skipped (not picked):', sorted({(s.get('category'), s.get('skip_reason')) for s in plan['skipped']}))
print(f'cubes: {len(cubes)} untouched check, max displacement {moved:.2f} mm')
if not result.get('task_complete'):
    raise SystemExit(f"Task incomplete: {result.get('abort_reason')}; inspect {report}")
if moved > 5.0:
    raise SystemExit(f'A cube moved {moved:.1f} mm; the cubes should be left alone.')
PY
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/frames" --out "$RUN/capvial_table.mp4" --fps 30
"$HOST_PYTHON" "$HERE/encode_video.py" --frames "$RUN/side_frames" --out "$RUN/capvial_side.mp4" --fps 30
echo "Review $RUN: videos, detections_overlay_yolo.png, sort_plan.json, execution_report.json"
