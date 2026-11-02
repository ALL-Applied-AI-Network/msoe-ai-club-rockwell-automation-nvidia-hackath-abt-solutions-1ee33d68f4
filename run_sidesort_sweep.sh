#!/usr/bin/env bash
# Run the clear-the-centre pipeline (tubes -y, cubes +y, caps stay) on N random layouts,
# one worker per GPU, then summarise.
#   ISAAC=... ./run_sidesort_sweep.sh [N=9] [OUT=runs/sidesort_sweep]
# Env: SEED0 (first layout seed, default 200), GPUS (default "0 1 2"), VIAL_RELEASE (default slide).
# Failures are recorded, not fatal: every layout is attempted and reported.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ISAAC="${ISAAC:-/isaacsim}"
HOST_PYTHON="${HOST_PYTHON:-$HERE/.venv/bin/python}"
YOLO_PYTHON="${YOLO_PYTHON:-$HERE/.venv-yolo/bin/python}"
N="${1:-9}"
OUT="${2:-$HERE/runs/sidesort_sweep}"
SEED0="${SEED0:-200}"
read -r -a GPUS <<< "${GPUS:-0 1 2}"
for executable in "$ISAAC/python.sh" "$HOST_PYTHON" "$YOLO_PYTHON"; do
    [ -x "$executable" ] || { echo "Launcher missing or not executable: $executable" >&2; exit 1; }
done
mkdir -p "$OUT"
[ -z "$(ls -A "$OUT")" ] || { echo "Use an empty output directory: $OUT" >&2; exit 1; }
"$HOST_PYTHON" "$HERE/capvial_layouts.py" --seeds $(seq "$SEED0" $((SEED0 + N - 1))) --out-dir "$OUT/layouts"

run_one() {
    local seed=$1 gpu=$2
    local run="$OUT/seed_$seed" layout="$OUT/layouts/layout_$seed.json"
    mkdir -p "$run"
    export CUDA_VISIBLE_DEVICES="$gpu"
    {
        "$ISAAC/python.sh" "$HERE/with_layout.py" "$layout" capture_scene.py --out "$run" &&
        "$HOST_PYTHON" "$HERE/perceive_opencv.py" --run "$run" &&
        "$YOLO_PYTHON" "$HERE/perceive_yolo.py" --run "$run" --merge &&
        "$HOST_PYTHON" "$HERE/plan_sidesort.py" --run "$run" &&
        "$ISAAC/python.sh" "$HERE/with_layout.py" "$layout" sort_capvial.py --run "$run" \
            --vial-release "${VIAL_RELEASE:-slide}"
    } > "$OUT/seed_$seed.log" 2>&1
    echo "seed $seed exit=$? gpu=$gpu" >> "$OUT/status.log"
    if [ -n "$(ls -A "$run/frames" 2>/dev/null)" ]; then
        "$HOST_PYTHON" "$HERE/encode_video.py" --frames "$run/frames" --out "$run/sidesort_table.mp4" --fps 30 >> "$OUT/seed_$seed.log" 2>&1
        "$HOST_PYTHON" "$HERE/encode_video.py" --frames "$run/side_frames" --out "$run/sidesort_side.mp4" --fps 30 >> "$OUT/seed_$seed.log" 2>&1
    fi
}

G=${#GPUS[@]}
for ((g = 0; g < G; g++)); do
    (
        for ((i = g; i < N; i += G)); do
            run_one $((SEED0 + i)) "${GPUS[g]}"
        done
    ) &
done
wait
"$HOST_PYTHON" "$HERE/summarize_sidesort_sweep.py" --dir "$OUT" | tee "$OUT/summary.txt"
