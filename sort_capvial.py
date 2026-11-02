"""Cap/vial executor: a copy of sort_scene.py kept separate for the cap/vial task.

The cube-stack executor (sort_scene.py) is left unchanged; this copy is where the
vial release strategy is developed and where per-frame release traces are recorded.
Run ./run_capvial_pipeline.sh RUN_DIR for the whole cap/vial pipeline.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scene_config as C
import fk
import ik
from cube_pose import square_yaw


# Vial release strategy (set from the command line); names combine, e.g. press_slide.
# baseline | press | wiggle | slide | press_wiggle | press_slide
VIAL_RELEASE = "baseline"
PRESS_M = 0.0015  # extra fingertip descent before opening (tip stays above the table)
WIGGLE_M = 0.0015  # sideways shuffle amplitude along the jaw direction after opening
SLIDE_M = 0.085  # travel along the tube axis: half its length (70 mm) plus finger clearance
SLIDE_SIGN = -1.0  # -1 slides toward the robot base (-x for tubes laid along x)
TRACE_SEGMENTS = (
    "carry", "lower", "press", "release", "settle",
    "wiggle_a", "wiggle_b", "wiggle_c", "slide", "retreat",
)


def ease(t):
    return t * t * (3 - 2 * t)


def turn(a, b, t):
    a, b = np.array(a, float), np.array(b, float)
    angle = np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a[:2], b[:2])) * t
    c, s = np.cos(angle), np.sin(angle)
    return np.array([a[0] * c - a[1] * s, a[0] * s + a[1] * c, 0.0])


def verify_cube_stack(positions, entries):
    """Check site, layer heights and actual adjacent-cube support geometry."""
    centres = []
    for entry in entries:
        name = entry.get("tracked_object")
        if name not in positions:
            return False
        pose = positions[name]
        if "yaw_deg" in pose and (
            abs(square_yaw(pose["yaw_deg"])) >= 8 or pose.get("tilt_deg", 0) >= 5
        ):
            return False
        centre = np.array(pose["centre"])
        if np.linalg.norm(centre[:2] - C.CUBE_STACK_XY) > C.CUBE_STACK_SITE_TOLERANCE:
            return False
        height = (
            C.TABLE_TOP_Z + C.SURFACE_DZ + (entry["stack_level"] + 0.5) * C.CUBE_EDGE
        )
        if abs(centre[2] - height) > 0.0015:
            return False
        if (
            centres
            and np.linalg.norm(centre[:2] - centres[-1][:2])
            > C.CUBE_STACK_LAYER_TOLERANCE
        ):
            return False
        centres.append(centre)
    return True


def align_signs(pick, q):
    """Pick the jaw-direction signs once for a grasp so approach/descend and
    carry/lower cannot differ by 180 degrees. Returns a copy of `pick`."""
    pick = dict(pick)
    if pick.get("spread_dir") is not None:
        r, _ = fk.fk(q)
        current = r @ np.array(C.FINGER_SPREAD_AXIS)
        spread = np.array(pick["spread_dir"], float)
        if np.dot(current, spread) < 0:
            spread = -spread
        pick["spread_dir"] = spread.tolist()
        if pick.get("place_spread_dir") is not None:
            destination = np.array(pick["place_spread_dir"], float)
            if np.dot(spread, destination) < 0:
                destination = -destination
            pick["place_spread_dir"] = destination.tolist()
    return pick


def segments(pick, home, look=None, start=None):
    """Motion segments for one pick. `look` = (tip target, jaw direction) prepends a
    wrist-camera look; the approach then starts from `start` (default `home`)."""
    p = np.array([*pick["xy"], pick["pick_z"]])
    d = np.array([*pick["place_xy"], pick["place_z"]])
    up, above = p + [0, 0, 0.13], d + [0, 0, 0.13]
    spread = pick.get("spread_dir")
    dest_spread = pick.get("place_spread_dir", spread)
    # Seat on the expected cube width instead of forcing rigid fingers through it.
    shut = C.CUBE_JAW_HALF_TARGET if pick["category"] == "cube" else C.JAW_CLOSED
    prefix = []
    if look is not None:
        look_tip, look_spread = look
        prefix = [
            ("look", home, look_tip, 1.2, C.JAW_OPEN, C.JAW_OPEN, look_spread),
            ("look_dwell", look_tip, look_tip, 0.5, C.JAW_OPEN, C.JAW_OPEN, look_spread),
        ]
        start = look_tip
    start = home if start is None else start
    release_at, pre, post = d, [], []
    if pick["category"] == "vial" and "press" in VIAL_RELEASE:
        # A lying tube is narrower nearer the table: set the fingertips lower before
        # opening, so the open fingers clear the tube instead of pinching it.
        release_at = d - [0, 0, PRESS_M]
        pre = [("press", d, release_at, 0.4, shut, shut, dest_spread)]
    open_ = C.JAW_OPEN
    if pick["category"] == "vial" and "wiggle" in VIAL_RELEASE:
        # Break static contact with whichever finger still touches the tube.
        side = np.array(dest_spread, float) * WIGGLE_M
        post = [
            ("wiggle_a", release_at, release_at + side, 0.2, open_, open_, dest_spread),
            ("wiggle_b", release_at + side, release_at - side, 0.4, open_, open_, dest_spread),
            ("wiggle_c", release_at - side, release_at, 0.2, open_, open_, dest_spread),
        ]
    retreat_from = release_at
    if pick["category"] == "vial" and "slide" in VIAL_RELEASE:
        # The open fingertip gap (~16 mm) is narrower than the tube (17 mm), so lifting
        # straight up drags the tube with the fingers. Slide the open gripper along the
        # tube's axis (perpendicular to the jaw direction) until the fingers clear its end.
        axis = np.array([dest_spread[1], -dest_spread[0], 0.0]) * SLIDE_SIGN
        retreat_from = release_at + axis * SLIDE_M
        post = post + [("slide", release_at, retreat_from, 0.8, open_, open_, dest_spread)]
    above_end = np.array([retreat_from[0], retreat_from[1], above[2]])
    return (
        prefix
        + [
            ("approach", start, up, 1.2, C.JAW_OPEN, C.JAW_OPEN, spread),
            ("descend", up, p, 0.8, C.JAW_OPEN, C.JAW_OPEN, spread),
            ("close", p, p, 0.6, C.JAW_OPEN, shut, spread),
            ("dwell", p, p, 0.5, shut, shut, spread),
            ("lift", p, up, 0.8, shut, shut, spread),
            ("carry", up, above, 1.4, shut, shut, dest_spread),
            ("lower", above, d, 0.8, shut, shut, dest_spread),
        ]
        + pre
        + [
            ("release", release_at, release_at, 0.6, shut, C.JAW_OPEN, dest_spread),
            ("settle", release_at, release_at, 0.6, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
        ]
        + post
        + [
            ("retreat", retreat_from, above_end, 0.8, C.JAW_OPEN, C.JAW_OPEN, dest_spread),
            ("home", above_end, home, 1.1, C.JAW_OPEN, C.JAW_OPEN, None),
        ]
    )


def validate_plan(plan):
    if plan.get("source") not in ("Cosmos3-Nano", "OpenCV"):
        raise ValueError(
            "Expected a Cosmos3-Nano or OpenCV sort plan; run the perception pipeline first"
        )
    if not np.isclose(plan.get("open_gap_m", 0), 2 * C.JAW_OPEN):
        raise ValueError(
            "Plan uses a different jaw opening; regenerate capture and perception"
        )
    if plan.get("task") == "cube_stack" and plan["picks"]:
        if [p.get("colour") for p in plan["picks"]] != list(C.CUBE_STACK_COLOURS):
            raise ValueError(
                "Tower must contain two yellow, two blue, then two red cubes"
            )
        if len({p["detection_id"] for p in plan["picks"]}) != len(C.CUBE_STACK_COLOURS):
            raise ValueError("Tower repeats a cube detection")
        for level, p in enumerate(plan["picks"]):
            yaw = float(p.get("yaw_deg", float("nan")))
            theta = np.deg2rad(yaw)
            expected = [-np.sin(theta), np.cos(theta), 0.0]
            spread = np.asarray(p.get("spread_dir", []), float)
            destination = np.asarray(p.get("place_spread_dir", []), float)
            if (
                not np.isfinite(yaw)
                or not np.isfinite(float(p.get("pose_confidence", 0)))
                or float(p.get("pose_confidence", 0)) < 0.8
                or spread.shape != (3,)
                or not np.allclose(spread, expected)
                or destination.shape != (3,)
                or not np.allclose(destination, [0.0, 1.0, 0.0])
            ):
                raise ValueError(
                    "Cube grasp requires a confident pose and aligned wrist"
                )
            if p["category"] != "cube" or p.get("stack_level") != level:
                raise ValueError("Invalid cube tower level")
            if not np.allclose(p["place_xy"], C.CUBE_STACK_XY) or not np.isclose(
                p["place_z"], C.SURFACE_DZ + level * C.CUBE_EDGE + C.CUBE_TIP_CLEARANCE
            ):
                raise ValueError("Invalid tower destination or placement height")
            if not np.isclose(p.get("pick_z", -1), C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE):
                raise ValueError(
                    "Cube grasp height differs from the current hooked-tip setup"
                )
            if not np.isclose(
                p.get("expected_centre_z", -1),
                C.SURFACE_DZ + (level + 0.5) * C.CUBE_EDGE,
            ):
                raise ValueError("Invalid expected tower height")
    for p in plan["picks"]:
        if p["category"] == "cap":
            if (
                p.get("orientation") != "narrow_end_up"
                or float(p.get("orientation_confidence", 0))
                < C.CAP_ORIENTATION_MIN_CONFIDENCE
            ):
                raise ValueError(
                    "Plan contains a cap without a confident narrow-end-up observation"
                )
            if not np.isclose(p["pick_z"], C.SURFACE_DZ + C.CAP_PICK_HEIGHT):
                raise ValueError("Cap pick height must clear the wide flange")
        if p["category"] == "vial":
            axis = np.array(p.get("axis_world", []), float)
            spread = np.array(p.get("spread_dir", []), float)
            if (
                axis.shape != (2,)
                or spread.shape != (3,)
                or not np.isclose(np.linalg.norm(spread), 1)
                or abs(np.dot(axis, spread[:2])) > 1e-5
            ):
                raise ValueError("Tube must have a perpendicular jaw closing direction")
        for key in ("xy", "place_xy"):
            if len(p[key]) != 2 or not np.all(np.isfinite(p[key])):
                raise ValueError("Invalid plan coordinates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        required=True,
        help="directory containing sort_plan.json and scene_state.json",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--camera", choices=["table", "side", "wrist"], default="table")
    ap.add_argument(
        "--physics-only",
        action="store_true",
        help="validate physical grasps without rendering frames",
    )
    ap.add_argument(
        "--no-wrist-look",
        action="store_true",
        help="skip the pre-grasp wrist-camera yaw refinement (cube_stack only)",
    )
    ap.add_argument(
        "--vial-release",
        choices=["baseline", "press", "wiggle", "slide", "press_wiggle", "press_slide"],
        default="baseline",
        help="how the gripper lets go of a set-down tube",
    )
    ap.add_argument("--press-mm", type=float, default=1.5)
    ap.add_argument("--wiggle-mm", type=float, default=1.5)
    ap.add_argument("--slide-mm", type=float, default=85.0)
    ap.add_argument("--slide-sign", type=float, default=-1.0, choices=[-1.0, 1.0])
    a = ap.parse_args()
    global VIAL_RELEASE, PRESS_M, WIGGLE_M, SLIDE_M, SLIDE_SIGN
    VIAL_RELEASE, PRESS_M, WIGGLE_M = a.vial_release, a.press_mm / 1000, a.wiggle_mm / 1000
    SLIDE_M, SLIDE_SIGN = a.slide_mm / 1000, a.slide_sign
    if a.fps != 30:
        ap.error("Use 30 fps: two 1/60 s physics steps per video frame")
    run = Path(a.run).resolve()
    out = Path(a.out).resolve() if a.out else run / "frames"
    if out.exists() and any(out.glob("*.png")):
        raise SystemExit(
            f"Output already contains PNGs: {out}. Choose a fresh run directory."
        )
    plan = json.loads((run / "sort_plan.json").read_text())
    validate_plan(plan)
    if not plan["picks"]:
        raise SystemExit(
            "No eligible picks; inspect detections.json and sort_plan.json before rendering"
        )
    state = json.loads((run / "scene_state.json").read_text())
    # Resolve paths and validate before Kit changes the working directory.
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import cv2
        import omni.usd
        from pxr import UsdPhysics
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        import cell
        import wrist_refine

        world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 30)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage)
        wrappers = {}
        for name, saved in state["objects"].items():
            prim = stage.GetPrimAtPath("/World/" + name)
            if not prim.IsValid():
                raise ValueError(f"Captured object missing: {name}")
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
            wrapper = SingleXFormPrim("/World/" + name)
            wrapper.set_world_pose(
                np.array(saved["position"]), np.array(saved["orientation"])
            )
            wrappers[name] = wrapper
        cam_paths = {
            "table": cell.TABLE_CAM_PATH,
            "side": cell.SIDE_CAM_PATH,
            "wrist": cell.WRIST_CAM_PATH,
        }
        configs = {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM}
        cam = (
            None
            if a.physics_only
            else Camera(
                prim_path=cam_paths[a.camera],
                resolution=configs[a.camera]["resolution"],
            )
        )
        # The look needs rendered wrist frames even when the video is not rendered.
        # Cube grasps are the yaw-sensitive ones: look before every cube pick, tower or not.
        wrist_look = (
            plan.get("task") == "cube_stack"
            or any(p["category"] == "cube" for p in plan["picks"])
        ) and not a.no_wrist_look
        wcam = (
            Camera(prim_path=cell.WRIST_CAM_PATH, resolution=C.WRIST_CAM["resolution"])
            if wrist_look
            else None
        )
        cell.enable_ccd(world)
        world.reset()
        if cam is not None:
            cam.initialize()
        if wcam is not None:
            wcam.initialize()
        arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        arm.initialize()
        names = list(arm.dof_names)
        arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        idx = {name: i for i, name in enumerate(names)}
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        tool = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
        home = np.array([0.30, 0.0, 0.24])
        down = [0.0, 0.0, -1.0]
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q, ok = ik.solve(q, home, down, limits, tool_offset=tool)
        if not ok:
            raise RuntimeError("Home IK failed")

        def command(q6, jaw):
            full = arm.get_joint_positions().copy()
            full[arm_idx] = q6
            for joint, value in cell.jaw_targets(jaw).items():
                full[idx[joint]] = value
            arm.apply_action(ArticulationAction(joint_positions=full))

        full = arm.get_joint_positions().copy()
        full[arm_idx] = q
        for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
            full[idx[joint]] = value
        arm.set_joint_positions(full)
        for _ in range(120):
            command(q, C.JAW_OPEN)
            world.step(render=not a.physics_only)
        for name in wrappers:
            UsdPhysics.RigidBodyAPI(
                stage.GetPrimAtPath("/World/" + name)
            ).CreateKinematicEnabledAttr().Set(False)
        for _ in range(30):
            command(q, C.JAW_OPEN)
            world.step(render=not a.physics_only)

        def poses():
            values = {}
            for name, wrapper in wrappers.items():
                pos, quat = wrapper.get_world_pose()
                w, x, y, z = quat
                rot = np.array(
                    [
                        [
                            1 - 2 * (y * y + z * z),
                            2 * (x * y - z * w),
                            2 * (x * z + y * w),
                        ],
                        [
                            2 * (x * y + z * w),
                            1 - 2 * (x * x + z * z),
                            2 * (y * z - x * w),
                        ],
                        [
                            2 * (x * z - y * w),
                            2 * (y * z + x * w),
                            1 - 2 * (x * x + y * y),
                        ],
                    ]
                )
                centre = np.array(pos)
                if name.startswith("SortVial"):
                    centre += rot @ np.array([0, 0, C.VIAL_LENGTH / 2])
                elif name.startswith("SortCap"):
                    centre += rot @ np.array([0, 0, C.CAP_HEIGHT / 2])
                elif name.startswith("SortCube"):
                    centre += rot @ np.array([0, 0, C.CUBE_EDGE / 2])
                values[name] = {
                    "position": pos.tolist(),
                    "orientation": quat.tolist(),
                    "centre": centre.tolist(),
                    "yaw_deg": float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))),
                    "tilt_deg": float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1)))),
                }
            return values

        before = poses()
        events = []
        frame = 0
        abort_reason = None
        out.mkdir(parents=True, exist_ok=True)
        # Additional fixed side camera provides a clear lift/contact review.
        side = None
        if not a.physics_only and a.camera != "side":
            side = Camera(
                prim_path=cell.SIDE_CAM_PATH, resolution=C.SIDE_CAM["resolution"]
            )
            side.initialize()
        side_out = run / "side_frames"
        side_out.mkdir(exist_ok=True)
        for pick in plan["picks"]:
            pick = align_signs(pick, q)
            start = poses()
            prefix = {"cap": "SortCap", "cube": "SortCube", "vial": "SortVial"}[
                pick["category"]
            ]
            candidates = [name for name in wrappers if name.startswith(prefix)]
            # This association is for execution reporting only, never motion commands.
            tracked = (
                min(
                    candidates,
                    key=lambda name: np.linalg.norm(
                        np.array(start[name]["centre"][:2]) - pick["xy"]
                    ),
                )
                if candidates
                else None
            )
            if (
                tracked
                and np.linalg.norm(np.array(start[tracked]["centre"][:2]) - pick["xy"])
                > 0.04
            ):
                tracked = None
            event = {
                "detection_id": pick["detection_id"],
                "category": pick["category"],
                "tracked_object": tracked,
                "pickup_verified": False,
                "placement_verified": False,
                "trajectory_ik_failures": 0,
            }
            if "stack_level" in pick:
                event.update(
                    stack_level=pick["stack_level"],
                    colour=pick["colour"],
                    perceived_yaw_deg=pick["yaw_deg"],
                    grasp_spread_dir=pick["spread_dir"],
                    place_spread_dir=pick["place_spread_dir"],
                )
            event["initial_positions"] = start
            look = None
            if wcam is not None and pick["category"] == "cube":
                look_tip = wrist_refine.look_target(pick["xy"], tool)
                _, look_ok = ik.solve(
                    q,
                    look_tip,
                    down,
                    limits,
                    tool_offset=tool,
                    spread_dir=wrist_refine.LOOK_SPREAD,
                    iters=400,
                )
                if look_ok:
                    look = (look_tip, np.array(wrist_refine.LOOK_SPREAD))
                else:  # keep the overhead yaw rather than fail a reachable grasp
                    event["wrist_look"] = {"ok": False, "reason": "look_pose_unreachable"}
            queue = segments(pick, home, look)
            trace = [] if pick["category"] == "vial" else None
            k = 0
            while k < len(queue):
                name, p0, p1, duration, j0, j1, spread = queue[k]
                k += 1
                spread_start = None
                if spread is not None and name in ("look", "approach", "carry"):
                    r, _ = fk.fk(q)
                    s = r @ np.array(C.FINGER_SPREAD_AXIS)
                    s[2] = 0
                    s /= np.linalg.norm(s)
                    spread = np.array(spread, float)
                    spread_start = s
                n = round(duration * a.fps)
                for i in range(n):
                    t = ease((i + 1) / n)
                    target = p0 + (p1 - p0) * t
                    direction = (
                        turn(spread_start, spread, t)
                        if spread_start is not None
                        else spread
                    )
                    q, ok = ik.solve(
                        q, target, down, limits, tool_offset=tool, spread_dir=direction
                    )
                    if not ok:
                        q, ok = ik.solve(
                            q,
                            target,
                            down,
                            limits,
                            tool_offset=tool,
                            spread_dir=direction,
                            iters=500,
                            damping=0.02,
                        )
                    r, actual = fk.tool_pose(q, tool)
                    aligned = (
                        direction is None
                        or abs(np.dot(r @ np.array(C.FINGER_SPREAD_AXIS), direction))
                        > 0.999
                    )
                    if not ok or np.linalg.norm(actual - target) > 0.001 or not aligned:
                        event["trajectory_ik_failures"] += 1
                        raise RuntimeError(
                            f'IK failed: detection {pick["detection_id"]}, {name}, frame {i}; refusing off-target descent'
                        )
                    command(q, j0 + (j1 - j0) * t)
                    for _ in range(2):
                        world.step(render=not a.physics_only)
                    if trace is not None and name in TRACE_SEGMENTS:
                        jp = arm.get_joint_positions()
                        trace.append(
                            {
                                "frame": frame,
                                "seg": name,
                                "t": round(t, 3),
                                "jaw_cmd": round(j0 + (j1 - j0) * t, 5),
                                "jaw_l": float(jp[idx["jaw_l"]]),
                                "jaw_r": float(jp[idx["jaw_r"]]),
                                "tip_z": float(actual[2]),
                                "obj": poses()[tracked]["centre"] if tracked else None,
                            }
                        )
                    if cam is not None:
                        rgba = cam.get_rgba()
                        if rgba is None:
                            raise RuntimeError("Camera returned no image")
                        cv2.imwrite(
                            str(out / f"f{frame:05d}.png"),
                            cv2.cvtColor(
                                rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                            ),
                        )
                    if side is not None:
                        rgba_side = side.get_rgba()
                        if rgba_side is not None:
                            cv2.imwrite(
                                str(side_out / f"f{frame:05d}.png"),
                                cv2.cvtColor(
                                    rgba_side[:, :, :3].astype(np.uint8),
                                    cv2.COLOR_RGB2BGR,
                                ),
                            )
                    frame += 1
                if name in ("descend", "dwell", "lift", "lower", "release"):
                    measured = arm.get_joint_positions()
                    rot, tip = fk.tool_pose(measured[arm_idx], tool)
                    measured_spread = rot @ np.array(C.FINGER_SPREAD_AXIS)
                    event.setdefault("phase_measurements", {})[name] = {
                        "actual_tip_robot_frame_m": tip.tolist(),
                        "actual_spread_dir": measured_spread.tolist(),
                        "commanded_spread_dir": (
                            np.asarray(direction).tolist()
                            if direction is not None
                            else None
                        ),
                        "wrist_joint_deg": float(np.degrees(measured[arm_idx][-1])),
                        "commanded_tip_robot_frame_m": target.tolist(),
                        "tip_error_mm": float(np.linalg.norm(tip - target) * 1000),
                        "object_centre_world_m": (
                            poses()[tracked]["centre"] if tracked else None
                        ),
                    }
                if name == "look_dwell":
                    frames = []
                    # The arm may have moved with rendering off (--physics-only), and
                    # the camera returns its last rendered buffer: let it catch up
                    # to the look pose first, or the frame shows the previous cube.
                    for _ in range(wrist_refine.RENDER_WARMUP_STEPS):
                        world.step(render=True)
                    for _ in range(wrist_refine.FRAMES):
                        rgba = None
                        for _ in range(20):  # the first renders may carry no data
                            world.step(render=True)
                            rgba = wcam.get_rgba()
                            if rgba is not None and rgba.size:
                                break
                        if rgba is None or not rgba.size:
                            raise RuntimeError("Wrist camera returned no image")
                        frames.append(
                            (
                                cv2.cvtColor(
                                    rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                                ),
                                wrist_refine.camera_model_from_stage(
                                    stage, cell.WRIST_CAM_PATH, C.WRIST_CAM["resolution"]
                                ),
                            )
                        )
                        for _ in range(wrist_refine.RENDER_WARMUP_STEPS // 4):
                            world.step(render=True)
                    look_dir = run / "wrist_look"
                    look_dir.mkdir(exist_ok=True)
                    cv2.imwrite(
                        str(look_dir / f'pick{pick["detection_id"]}.png'), frames[0][0]
                    )
                    refined = wrist_refine.refine_yaw(frames, pick["yaw_deg"], pick["xy"])
                    event["wrist_look"] = refined
                    print(
                        f'[sort] detection {pick["detection_id"]}: wrist yaw {refined["yaw"]} '
                        f'(overhead {pick["yaw_deg"]:.1f}); ok={refined["ok"]} {refined["reason"] or ""}',
                        flush=True,
                    )
                    if not refined["ok"]:
                        event["wrist_abort"] = (
                            f'Wrist look rejected detection {pick["detection_id"]}: {refined["reason"]}'
                        )
                        break
                    theta = np.deg2rad(refined["yaw"])
                    pick["yaw_deg"] = refined["yaw"]
                    pick["spread_dir"] = [-float(np.sin(theta)), float(np.cos(theta)), 0.0]
                    pick = align_signs(pick, q)
                    event["grasp_spread_dir"] = pick["spread_dir"]
                    queue[k:] = segments(pick, home, start=p1)
                if name == "dwell":
                    jp = arm.get_joint_positions()
                    event["jaw_gap_mm"] = float(
                        (2 * C.JAW_OPEN - abs(jp[idx["jaw_l"]]) - abs(jp[idx["jaw_r"]]))
                        * 1000
                    )
                if name == "lift":
                    lifted = poses()
                    event["objects_lifted"] = [
                        key
                        for key in wrappers
                        if lifted[key]["centre"][2] - start[key]["centre"][2] > 0.025
                    ]
                    event["pickup_verified"] = tracked in event["objects_lifted"]
                    print(
                        f'[sort] detection {pick["detection_id"]}: lifted {event["objects_lifted"]}; pickup={event["pickup_verified"]}',
                        flush=True,
                    )
            finish = poses()
            if tracked:
                position = np.array(finish[tracked]["centre"])
                event["placement_error_mm"] = float(
                    np.linalg.norm(position[:2] - pick["place_xy"]) * 1000
                )
                expected_z = (
                    C.TABLE_TOP_Z
                    + C.SURFACE_DZ
                    + {
                        # 1.5 edges = sitting on another cube; a side-slot cube lies on the table
                        "cube": (0.5 if "side_slot" in pick else 1.5) * C.CUBE_EDGE,
                        "cap": C.CAP_HEIGHT / 2,
                        "vial": C.VIAL_RADIUS,
                    }[pick["category"]]
                )
                if "stack_level" in pick:
                    expected_z = C.TABLE_TOP_Z + pick["expected_centre_z"]
                event["placement_verified"] = bool(
                    event["pickup_verified"]
                    and event["placement_error_mm"] < 20
                    and abs(position[2] - expected_z) < 0.01
                )
                if "side_slot" in pick:  # a cube set down flat at its slot
                    event["placement_tilt_deg"] = finish[tracked]["tilt_deg"]
                    event["placement_verified"] = bool(
                        event["pickup_verified"]
                        and event["placement_error_mm"] < 8
                        and abs(position[2] - expected_z) < 0.003
                        and event["placement_tilt_deg"] < 10
                    )
                if "stack_level" in pick:
                    event["placement_verified"] = bool(
                        event["pickup_verified"]
                        and event["placement_error_mm"]
                        < C.CUBE_STACK_SITE_TOLERANCE * 1000
                        and abs(position[2] - expected_z) < 0.0015
                    )
                    event["placement_yaw_error_deg_mod90"] = abs(
                        square_yaw(finish[tracked]["yaw_deg"])
                    )
                    event["placement_tilt_deg"] = finish[tracked]["tilt_deg"]
                    event["placement_pose_verified"] = bool(
                        event["placement_yaw_error_deg_mod90"] < 8
                        and event["placement_tilt_deg"] < 5
                    )
                    event["placement_verified"] = (
                        event["placement_verified"] and event["placement_pose_verified"]
                    )
                    event["stack_support_verified"] = verify_cube_stack(
                        finish, events + [event]
                    )
            if trace:
                event["release_trace"] = trace
            events.append(event)
            (run / "execution_events.json").write_text(
                json.dumps(events, indent=2) + "\n"
            )
            if plan.get("task") == "cube_stack" and (
                not event["placement_verified"]
                or not event.get("stack_support_verified")
            ):
                abort_reason = event.get(
                    "wrist_abort"
                ) or f'Cube tower stopped at level {pick["stack_level"]}: pickup, placement or supporting stack failed'
                print(f"[sort] {abort_reason}", flush=True)
                break
        after = poses()
        report = {
            "before": before,
            "after": after,
            "picks": events,
            "frames": frame,
            "rendered": not a.physics_only,
            "task": plan.get("task", "sort"),
            "abort_reason": abort_reason,
            "hardware": {
                "gripper_model": C.GRIPPER_MODEL,
                "jaw_stroke_m": C.JAW_STROKE,
                "open_gap_m": 2 * C.JAW_OPEN,
                "finger_length_m": C.FINGER_LENGTH,
                "cube_edge_m": C.CUBE_EDGE,
                "cube_tip_clearance_m": C.CUBE_TIP_CLEARANCE,
            },
            "task_complete": bool(plan["picks"])
            and abort_reason is None
            and len(events) == len(plan["picks"])
            and all(e["placement_verified"] for e in events),
            "verified_pickups": sum(e["pickup_verified"] for e in events),
            "verified_placements": sum(e["placement_verified"] for e in events),
        }
        # Evaluate untouched wide-up objects only after executing the vision plan.
        gt = json.loads((run / "ground_truth.json").read_text())
        wide = [
            g["name"] for g in gt["objects"] if g.get("orientation") == "wide_end_up"
        ]
        report["wide_caps"] = [
            {
                "name": name,
                "displacement_mm": float(
                    np.linalg.norm(
                        np.array(after[name]["centre"]) - before[name]["centre"]
                    )
                    * 1000
                ),
                "picked_up": any(name in e.get("objects_lifted", []) for e in events),
            }
            for name in wide
        ]
        (run / "execution_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f'[sort] {report["verified_pickups"]}/{len(events)} pickups; {report["verified_placements"]} verified placements; {frame} frames',
            flush=True,
        )
    except Exception:
        # Kit shutdown may terminate Python before the exception is displayed.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
