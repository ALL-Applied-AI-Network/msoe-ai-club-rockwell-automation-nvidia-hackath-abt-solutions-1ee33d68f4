# ABT Solutions - FANUC Hackathon Sorting Solution

This minimal starter contains code, instructions, tests and required geometry.
**No images, videos, saved runs, model weights or virtual environments are in the repository.**

Two perception paths exist side by side: the original **Cosmos3-Nano** pipeline
(`run_sort_pipeline.sh`) and a **Cosmos-free** one that needs no Cosmos weights: OpenCV for the
cubes (no weights at all) and a small YOLO-OBB detector, trained on synthetic renders, for caps
and tubes. The detector weights are not in git; download them from the GitHub Release or train
them (SETUP.md 3a).

- People: follow [SETUP.md](SETUP.md).
- Coding agents: read [AGENTS.md](AGENTS.md), then follow the same setup guide.
- Physical-system calibration: [docs/CALIBRATION_V05.md](docs/CALIBRATION_V05.md).
- Geometry/perception assumptions: [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md).

```bash
./bootstrap.sh
# Configure ISAAC, COSMOS and model paths as described in SETUP.md.
./run_sort_pipeline.sh runs/stack_01          # Cosmos path

# Cosmos-free path (SETUP.md sections 3a-4a):
./run_opencv_pipeline.sh runs/cubes_01        # cube tower, OpenCV + wrist-camera yaw
./run_capvial_pipeline.sh runs/capvial_01     # tubes aside, cubes left alone (needs YOLO weights)
./run_sidesort_pipeline.sh runs/sidesort_01   # tubes -y, cubes +y, caps stay
```

Bootstrap installs host dependencies, generates procedural material textures
locally and runs host tests. It does not install Isaac Sim or Cosmos. Detailed
finger geometry is retained; photo-derived finishes and serial markings are
replaced by a silver material when the optional photo maps are absent.

## Architecture and entry points

| File | Responsibility |
| --- | --- |
| `scene_config.py` | Dimensions, cube positions/rotations, stack order and camera configuration |
| `cell.py`, `gripper_geometry.py` | USD scene, gripper geometry and contact physics |
| `perceive.py`, `cube_pose.py` | Cosmos semantics and calibrated image-derived tabletop pose |
| `plan_sort.py`, `backproject.py`, `plan_grasps.py` | Vision-only planning and image/camera geometry |
| `sort_scene.py`, `ik.py`, `fk.py` | Physical execution, wrist alignment and success checks |
| `build_scene.py` | Interactive scene or headless build/render |
| `run_sort_pipeline.sh` | Capture → perceive → plan → execute → encode (Cosmos) |
| `perceive_opencv.py`, `wrist_refine.py` | Cosmos-free cube detection, noise normalisation, wrist-camera yaw refinement |
| `gen_detector_data.py`, `augment_detector_data.py`, `draw_labels.py`, `train_detector.py`, `perceive_yolo.py`, `capvial_detections.py` | Synthetic data, corrupted-copy augmentation, label check, training, inference and detection formatting for the cap/tube YOLO-OBB detector |
| `eval_detector.py`, `holdout_eval.py`, `perturb_eval.py` | Detector scoring (clean, corrupted, unseen corruptions) and OpenCV cube stress tests; ground truth used for evaluation only |
| `plan_capvial.py`, `plan_sidesort.py`, `sort_capvial.py`, `capvial_config.py` | Side-slot planning and execution for tubes and cubes (cap/tube and clear-the-centre tasks) |
| `run_opencv_pipeline.sh`, `run_capvial_pipeline.sh`, `run_sidesort_pipeline.sh`, `run_*_sweep.sh` | Cosmos-free launchers and random-layout sweeps |
| `capvial_layouts.py`, `with_layout.py`, `summarize_*_sweep.py` | Seeded random layouts, scene patching for a run, sweep summaries |
| `test_*.py` | Host tests (`bootstrap.sh` runs 69) |

On the Cosmos path, Cosmos identifies cube locations and colours. A calibrated image-geometry
front end measures yaw; Cosmos receives that measurement and confirms/reports
the pose. Standalone VLM angle estimation was unreliable. This hybrid system
uses no simulator ground-truth object poses to command grasps.

The actuator has **6 mm travel per jaw**, giving a **12 mm maximum straight-face
gap**. Square yaw is equivalent every 90°. Keep these constraints when editing.
Changes to layout, camera, cube size or contact geometry need a fresh run.

Rebuild this media-free package with `./share.sh --output ../starter.zip`.

## Cosmos-free results and limits

Simulation only, small samples (9 random layouts per sweep, all passed):

| Task | Layouts complete | Placement error |
| --- | --- | --- |
| Tubes set aside, cubes left alone | 9/9 | tubes mean 1.6 mm |
| Clear the centre (tubes -y, cubes +y, caps stay) | 9/9, caps never moved | cubes mean 0.1 mm, tubes mean 2.1 mm |

- Cube yaw is refined by a wrist-camera look before each pick; gates reject a cube rather than force a bad grasp.
- Detector: the base weights are weak on dark, dim, noisy and JPEG images. The `capvial_n_aug` fine-tune
  fixes this in detector-only tests, including corruptions it was not trained on, but it has not yet been
  run through full pipeline sweeps. Bright (1.4x) images were not trained for.
- The Cosmos-path files were left in working order, with small additions only: `plan_sort.py` (`--image`,
  source label, top-face yaw agreement gate), `sort_scene.py` (accepts OpenCV detections, optional wrist
  look, `--no-wrist-look`) and `bootstrap.sh` (more tests). The Cosmos path itself has not been re-run
  since these edits; only host tests cover it.
