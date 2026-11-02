"""Synthetic training images for the cap/vial detector (runs under Isaac's interpreter).

Renders the calibrated table camera at its native 1920x1080 over many random layouts
and writes YOLO-OBB labels computed exactly from the known object poses. Cubes are
rendered as unlabelled distractors (the OpenCV pipeline detects those).

    $ISAAC/python.sh gen_detector_data.py --out datasets/capvial --split train --n 600 --seed 1

Classes: 0 cap_narrow_up, 1 cap_wide_up, 2 vial.
Ground-truth poses are written beside each image for evaluation only.
"""

import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--split", default="train")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--absent-prob", type=float, default=0.25, help="chance an object is left off the table")
args = parser.parse_args()
args.out = os.path.abspath(args.out)  # Kit changes the working directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C

N_CAPS, N_VIALS = 4, 3
# Four caps (two of each orientation) and three vials; poses are overwritten per image.
C.SORT_CAPS = [(0.30 + 0.05 * i, 0.0) for i in range(N_CAPS)]
C.CAP_ORIENTATIONS = ["narrow_end_up", "narrow_end_up", "wide_end_up", "wide_end_up"]
C.SORT_VIALS = [(0.30 + 0.05 * i, 0.1 * i, 10.0 + i) for i in range(N_VIALS)]

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import cv2
import numpy as np
import omni.usd
from pxr import UsdPhysics
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera

import cell
from wrist_refine import camera_model_from_stage

CLASSES = ["cap_narrow_up", "cap_wide_up", "vial"]
Z = C.TABLE_TOP_Z + C.SURFACE_DZ
X_RANGE, Y_RANGE = (0.20, 0.62), (-0.32, 0.32)
FAR = np.array([10.0, 10.0, Z])  # parking spot for "absent" objects, far outside the view
MARGIN = 0.012
CUBE_R, CAP_R, VIAL_R = 0.009, C.CAP_RADIUS, C.VIAL_RADIUS


def quat_wxyz(rotation):
    x, y, z, w = rotation.as_quat()
    return np.array([w, x, y, z])


def footprint(kind, xy, yaw):
    """Sample points and radius approximating an object's table footprint."""
    if kind == "vial":
        axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        t = np.linspace(-C.VIAL_LENGTH / 2, C.VIAL_LENGTH / 2, 15)
        return np.asarray(xy) + t[:, None] * axis, VIAL_R
    return np.asarray(xy)[None, :], (CAP_R if kind == "cap" else CUBE_R)


def clear(a, b):
    (pa, ra), (pb, rb) = a, b
    gap = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min()
    return gap >= ra + rb + MARGIN


def sample_layout(rng):
    kinds = ["cube"] * len(C.SORT_CUBES) + ["cap"] * N_CAPS + ["vial"] * N_VIALS
    placed, layout = [], []
    for kind in kinds:
        if kind != "cube" and rng.random() < args.absent_prob:
            layout.append(dict(kind=kind, present=False))
            continue
        for _ in range(2000):
            xy = (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE))
            yaw = rng.uniform(-90, 90)
            fp = footprint(kind, xy, yaw)
            near_tower = np.linalg.norm(np.asarray(xy) - C.CUBE_STACK_XY) < 0.05
            if not near_tower and all(clear(fp, other) for other in placed):
                placed.append(fp)
                layout.append(dict(kind=kind, present=True, xy=xy, yaw=yaw))
                break
        else:
            layout.append(dict(kind=kind, present=False))
    return layout


def world_pose(kind, xy, yaw):
    if kind == "vial":  # origin on the axis; long axis is local +Z, rotated flat
        axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        position = np.array([*np.asarray(xy), Z + VIAL_R + 0.001])
        rot = Rotation.from_euler("z", yaw, degrees=True) * Rotation.from_euler("y", 90, degrees=True)
        # cell.py places the xform origin at the vial's tail, half a length back
        position[:2] = np.asarray(xy) - C.VIAL_LENGTH / 2 * axis
        return position, quat_wxyz(rot)
    return np.array([*xy, Z]), quat_wxyz(Rotation.from_euler("z", yaw, degrees=True))


def project_hull(cam, points):
    pts = np.array([cam.project(p) for p in points], np.float32)
    return cv2.convexHull(pts).reshape(-1, 2)


def cap_label(cam, xy):
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], 1) * CAP_R
    points = [(xy[0] + dx, xy[1] + dy, Z + z) for z in (0.0, C.CAP_HEIGHT) for dx, dy in ring]
    hull = project_hull(cam, points)
    (x0, y0), (x1, y1) = hull.min(0) - 1, hull.max(0) + 1
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def vial_label(cam, xy, yaw):
    axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
    side = np.array([-axis[1], axis[0]])
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    zc = Z + VIAL_R + 0.001
    points = []
    for end in (-1, 1):
        centre = np.asarray(xy) + end * C.VIAL_LENGTH / 2 * axis
        for a in angles:
            off = VIAL_R * np.cos(a) * side
            points.append((*(centre + off), zc + VIAL_R * np.sin(a)))
    box = cv2.boxPoints(cv2.minAreaRect(project_hull(cam, points)))
    ends = [cam.project([*(np.asarray(xy) + e * C.VIAL_LENGTH / 2 * axis), zc]) for e in (-1, 1)]
    return box.astype(np.float32), np.array(ends)


def main():
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage)
    for prim in stage.Traverse():
        if prim.GetName().startswith(("SortCube", "SortVial", "SortCap")) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)
    cam = Camera(prim_path=cell.TABLE_CAM_PATH, resolution=C.TABLE_CAM["resolution"])
    cell.enable_ccd(world)
    world.reset()
    cam.initialize()
    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    q = arm.get_joint_positions()
    for joint, deg in C.CAPTURE_POSE_DEG.items():  # park the arm out of the table view
        q[names.index(joint)] = np.deg2rad(deg)
    for joint, value in cell.jaw_targets(C.JAW_OPEN).items():
        q[names.index(joint)] = value
    arm.set_joint_positions(q)
    arm.apply_action(ArticulationAction(joint_positions=q))

    prims = {
        "cube": [SingleXFormPrim(f"/World/SortCube{i}") for i in range(len(C.SORT_CUBES))],
        "cap": [SingleXFormPrim(f"/World/SortCap{i}") for i in range(N_CAPS)],
        "vial": [SingleXFormPrim(f"/World/SortVial{i}") for i in range(N_VIALS)],
    }
    lights = [stage.GetPrimAtPath(p) for p in ("/World/DomeLight",)]
    base_intensity = [float(p.GetAttribute("inputs:intensity").Get()) for p in lights]
    for _ in range(90):  # let materials compile
        world.step(render=True)
    cam_model = camera_model_from_stage(stage, cell.TABLE_CAM_PATH, C.TABLE_CAM["resolution"])
    W, H = C.TABLE_CAM["resolution"]

    for sub in ("images", "labels", "meta"):
        os.makedirs(os.path.join(args.out, sub, args.split), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for n in range(args.n):
        layout = sample_layout(rng)
        counters = {"cube": 0, "cap": 0, "vial": 0}
        labels, meta = [], []
        for item in layout:
            kind = item["kind"]
            prim = prims[kind][counters[kind]]
            index = counters[kind]
            counters[kind] += 1
            if not item["present"]:
                prim.set_world_pose(FAR, np.array([1.0, 0, 0, 0]))
                continue
            position, orientation = world_pose(kind, item["xy"], item["yaw"])
            prim.set_world_pose(position, orientation)
            if kind == "cap":
                cls = 0 if C.CAP_ORIENTATIONS[index] == "narrow_end_up" else 1
                box = cap_label(cam_model, item["xy"])
                meta.append(dict(cls=CLASSES[cls], xy=list(item["xy"]), yaw=item["yaw"]))
            elif kind == "vial":
                cls = 2
                box, ends = vial_label(cam_model, item["xy"], item["yaw"])
                meta.append(dict(cls="vial", xy=list(item["xy"]), yaw=item["yaw"], axis_endpoints_px=ends.tolist()))
            else:
                continue
            if box[:, 0].min() < 0 or box[:, 1].min() < 0 or box[:, 0].max() > W or box[:, 1].max() > H:
                continue  # partly out of frame: leave unlabelled rather than mislabel
            norm = (box / [W, H]).reshape(-1)
            labels.append(f"{cls} " + " ".join(f"{v:.6f}" for v in norm))
        scale = rng.uniform(0.5, 1.6)
        for light, base in zip(lights, base_intensity):
            light.GetAttribute("inputs:intensity").Set(base * scale)
        for _ in range(12):  # render lags pose changes; read only after it catches up
            world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is None or not rgba.size:
            raise RuntimeError("Camera returned no image")
        name = f"{args.split}_{args.seed}_{n:05d}"
        cv2.imwrite(os.path.join(args.out, "images", args.split, name + ".png"), cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        with open(os.path.join(args.out, "labels", args.split, name + ".txt"), "w") as f:
            f.write("\n".join(labels) + ("\n" if labels else ""))
        with open(os.path.join(args.out, "meta", args.split, name + ".json"), "w") as f:
            json.dump(dict(light_scale=scale, objects=meta), f)
        print(f"[gen] {name}: {len(labels)} labelled objects", flush=True)
    sim_app.close()


main()
