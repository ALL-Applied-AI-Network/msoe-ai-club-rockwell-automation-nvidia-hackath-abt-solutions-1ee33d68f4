# Setup checklist for people and coding agents

There are two perception paths. **Cosmos-free (OpenCV + YOLO)** needs no model
weights and is the path validated in this repository (sections 3a-4a). **Cosmos3-Nano**
is the original path (sections 3-4) and is unchanged. Both share section 1-2.

Run commands from the extracted `fanuc_er4ia_sim/` directory. Replace all
`/path/to/...` placeholders with actual machine paths. The starter has no media
or saved run to replay. It was validated on Linux with Isaac Sim 6.0 and an RTX
A6000; Windows and smaller GPUs have not been benchmarked.

## 1. Host setup

Prerequisites: Python with `venv` and pip. Host Python 3.12 was used for validation.

```bash
./bootstrap.sh
```

This creates/reuses `.venv`, installs `requirements.txt`, generates procedural
metal/environment maps in `textures/`, and runs 69 host tests. No reference images
are needed. Detailed finger meshes use silver materials without photo maps.
These generated maps are local outputs, not bundled image files.

## 2. Isaac Sim

Install Isaac Sim separately, or reuse an existing installation:

```bash
export ISAAC=/path/to/isaacsim
mkdir -p runs
"$ISAAC/python.sh" build_scene.py --headless --save runs/scene.usd
```

The scene loads the FANUC ER-4iA robot USD from Isaac's asset root. Network access
or an available asset cache is needed on first use. Robot USD and its referenced
files are not bundled. An optional local asset/URI override is supported:

```bash
export FANUC_ROBOT_USD=/path/to/er4ia.usd
```

Preserve the asset's dependencies, original joint names and hierarchy.
To open the scene interactively, run `"$ISAAC/python.sh" build_scene.py`.

## 3. Cosmos3-Nano (skip if using the Cosmos-free path)

Use a separate environment with CUDA-enabled PyTorch, compatible Transformers,
and NVIDIA's `transformers-cosmos3` package/framework. Obtain Cosmos3-Nano model
weights separately; these large dependencies are not in the starter.

```bash
export COSMOS=/path/to/cosmos-environment/bin/python
"$COSMOS" -m pip install -r requirements-cosmos.txt
export COSMOS3_NANO_PATH=/path/to/Cosmos3-Nano/snapshot
```

If the environment uses uv instead of pip:
`uv pip install --python "$COSMOS" -r requirements-cosmos.txt`.
The snapshot must contain `config.json` and the model/processor files.
If the framework package is not installed, set its checkout location:

```bash
export COSMOS3_ROOT=/path/to/cosmos-framework
```

That directory must contain `packages/transformers-cosmos3/transformers_cosmos3/`.
With no explicit snapshot, model discovery uses the Hugging Face cache;
`HF_HOME` or `HF_HUB_CACHE` can select its location.

Check dependencies and model loading before a complete run:

```bash
"$COSMOS" -c 'import torch, transformers, scipy, cv2; print("CUDA:", torch.cuda.is_available())'
"$COSMOS" -c 'from perceive import load_model; model, processor = load_model(); print("Model ready")'
```

CUDA must be available. The second command also checks framework imports and
weights, and exits after loading to release GPU memory.

## 4. Run and verify

```bash
./run_sort_pipeline.sh runs/stack_01
```

Use an empty run directory. Without an argument, the launcher creates a
timestamped directory. `HOST_PYTHON` optionally overrides `.venv/bin/python`.

| Stage | Interpreter | Entry point |
| --- | --- | --- |
| Calibrated 4K table capture | Isaac | `capture_scene.py` |
| Cube semantics and measured pose reporting | Cosmos | `perceive.py --stack-cubes` |
| Plan and evaluation | Host | `plan_sort.py` |
| Physical grasps and wrist control | Isaac | `sort_scene.py` |
| Locally generated review videos | Host | `encode_video.py` |

The run produces its own images, videos and JSON reports. Read
`execution_report.json` and require `task_complete: true`, six verified pickups
and six verified placements. Inspect `sort_plan.json` for skips and
`perception_evaluation.json` for position/colour/yaw errors. Raw model replies
and image-measured pose inputs are saved under `cube_pose_crops/`.
Square yaw errors use 90° symmetry. Ground truth evaluates results only.

To rerun physical validation without rendering, use a fresh captured/planned run:

```bash
"$ISAAC/python.sh" sort_scene.py --run runs/new_planned_run --physics-only
```

## 3a. Cosmos-free perception (OpenCV cubes, YOLO caps and vials)

Cubes are found by OpenCV (`perceive_opencv.py`: colour blobs, noise/background
normalisation, silhouette and top-face yaw). Caps and tubes are found by a small
YOLOv8n-OBB detector (`perceive_yolo.py`) trained on synthetic renders. Cube-only
runs need just the host `.venv`; the cap/vial tasks also need the YOLO environment:

```bash
python3 -m venv .venv-yolo && .venv-yolo/bin/pip install -r requirements-yolo.txt
```

No trained weights are stored in git (`*.pt` is ignored). Either download `capvial_n_aug_best.pt` from the
repository's GitHub Release (checksums in `SHA256SUMS`) and place it at
`runs_detector/capvial_n_aug/weights/best.pt`, or generate data and train (about 700 renders):

```bash
"$ISAAC/python.sh" gen_detector_data.py --out datasets/capvial --split train --n 600 --seed 1
"$ISAAC/python.sh" gen_detector_data.py --out datasets/capvial --split val --n 100 --seed 2
.venv/bin/python draw_labels.py --dataset datasets/capvial --split train --out sheet.png   # eyeball labels
.venv-yolo/bin/python train_detector.py --dataset datasets/capvial --name capvial_n
.venv-yolo/bin/python eval_detector.py --weights runs_detector/capvial_n/weights/best.pt \
    --dataset datasets/capvial --split val --camera runs/<any_run>/camera.json
```

To harden the detector against dark, dim, noisy and JPEG-compressed images, build a corrupted copy
of the training set and fine-tune from the first model (about 12 minutes on one T4):

```bash
.venv/bin/python augment_detector_data.py --src datasets/capvial --out datasets/capvial_aug
.venv-yolo/bin/python train_detector.py --dataset datasets/capvial_aug --name capvial_n_aug \
    --model runs_detector/capvial_n/weights/best.pt --epochs 20
.venv-yolo/bin/python holdout_eval.py --weights runs_detector/capvial_n_aug/weights/best.pt \
    --dataset datasets/capvial --split val --camera runs/<any_run>/camera.json   # unseen corruptions
```

Then pass `--weights runs_detector/capvial_n_aug/weights/best.pt` to `perceive_yolo.py`
(or change its default).

`perceive_yolo.py` loads `runs_detector/capvial_n/weights/best.pt` by default
(`--weights` overrides). Ground truth is used only by the evaluation scripts.

## 4a. Run the Cosmos-free pipelines

Each launcher needs `ISAAC` set and an empty run directory (or none, for a timestamped one).
They are independent of each other and of the Cosmos launcher.

| Launcher | Task |
| --- | --- |
| `run_opencv_pipeline.sh` | Cube tower (yellow, blue, red), OpenCV only, with wrist-camera yaw look |
| `run_capvial_pipeline.sh` | Tubes set aside at the -y side; cubes left alone |
| `run_sidesort_pipeline.sh` | Clear the centre: tubes to -y, cubes to +y, caps stay |
| `run_capvial_sweep.sh`, `run_sidesort_sweep.sh` | N seeded random layouts, one worker per GPU (`GPUS="0 1 2"`), then a summary |

```bash
export ISAAC=/path/to/isaacsim
./run_opencv_pipeline.sh runs/cubes_01
./run_sidesort_pipeline.sh runs/sidesort_01
./run_sidesort_sweep.sh 9 runs/sidesort_sweep      # summary.txt in that directory
```

Read `execution_report.json` for `task_complete: true` and verified pickups/placements.
Quality gates are unchanged (pose confidence, silhouette IoU, yaw-estimator agreement,
wrist-look agreement). A cube that fails a gate is skipped, not forced.

Behaviour to know about:

- **Caps are never picked.** The cap plug (23.6 mm) exceeds the 12 mm finger opening, so
  the planner skips them (`plug_exceeds_opening`). Moving caps needs new fingers.
- **Tubes** are released by sliding the open gripper 85 mm along the tube axis and then
  lifting (`--vial-release slide`, the default); pressing down or wiggling did not free them.
- **Wrist look**: before each cube pick the arm images the cube from the wrist camera to
  refine yaw (about 1 degree vs 3-5 degrees from overhead). `--no-wrist-look` disables it.
- **Robustness**: `perturb_eval.py --run <run>` re-detects under noise, lighting, blur, JPEG
  and table recolouring. The base YOLO weights are weak on a very dark table, 0.3x lighting and
  JPEG compression; the `capvial_n_aug` fine-tune fixes these in detector-only tests (not yet in full pipeline runs).
- Results so far are simulation only: 9/9 random layouts for both cap/vial and
  clear-the-centre sweeps (cube slot error mean 0.1 mm, tube 2.1 mm), small samples.
- Layout experiments: `with_layout.py <layout.json> <script> ...` patches `scene_config`
  lists before running a script; `capvial_layouts.py` generates seeded layouts.

## Troubleshooting

- Missing launcher: check the interpreter paths and executable permissions.
- Missing model/framework: verify the snapshot and package/checkout paths.
- Missing robot: check Isaac's asset root or `FANUC_ROBOT_USD` and dependencies.
- YOLO weights missing: train them (section 3a) or pass `--weights`.
- Black/stale wrist frames, `wrist_abort`: expected fail-safe; inspect `wrist_look/`.
- No eligible picks: inspect skips and raw model replies; all six poses must pass.
- Incomplete stack: inspect the execution report and simulator logs.
- Changed geometry/layout/camera: recapture and replan in an empty directory.

Agents should report missing dependencies and completed checks precisely;
host tests do not prove model or simulator execution. Read `AGENTS.md` for
editing constraints and success criteria.
