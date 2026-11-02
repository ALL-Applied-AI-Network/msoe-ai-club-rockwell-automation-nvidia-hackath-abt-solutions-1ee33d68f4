"""Wrist-camera yaw refinement for the cube-stack executor.

The overhead camera sees a cube as a ~30 px blob, so its yaw is good to only a few
degrees; the gripper tolerates roughly 4 deg. Before each grasp the arm hovers with
the wrist camera (not the tool) above the cube, where the cube is ~40 px across and
seen straight down, and the yaw is re-measured from the top face.

Pure geometry/image functions: usable under Isaac's interpreter and on the host.
"""

import cv2
import numpy as np

import scene_config as C
from backproject import CameraModel
from cube_pose import square_yaw
from perceive_opencv import TABLE_CHROMA_TOL, TABLE_TARGET, top_face_yaw

LOOK_SPREAD = (0.0, 1.0, 0.0)  # jaw direction during the look; fixes the camera offset
LOOK_TIP_CLEARANCE = 0.045  # fingertip height above the work surface (caps are 16 mm)
FRAMES = 3  # frames measured per look; the median is used
RENDER_WARMUP_STEPS = 12  # render steps before the first capture (probe: stale after ~1)
WRIST_AGREEMENT_DEG = 10.0  # max wrist vs overhead yaw disagreement (gross errors)
MAX_FRAME_SPREAD_DEG = 3.0  # max spread across the frames of one look
NOISE_FLOOR = 2.0  # estimated luminance noise (grey levels) below which nothing is denoised


def look_target(xy, tool_vec):
    """Tool-tip target (robot frame) putting the wrist camera's axis above `xy`.

    The look orientation is fixed (tool axis down, jaws along LOOK_SPREAD), so the
    camera's offset from the fingertip is one constant world vector; the lens sits
    ~10 cm off the tool axis and would otherwise leave the cube at the frame edge.
    """
    tx, ty = np.array(C.TOOL_AXIS, float), np.array(C.FINGER_SPREAD_AXIS, float)
    local = np.stack([tx, ty, np.cross(tx, ty)], axis=1)
    x_des = np.array([0.0, 0.0, -1.0])
    y_des = np.array(LOOK_SPREAD, float)
    world = np.stack([x_des, y_des, np.cross(x_des, y_des)], axis=1)
    offset = world @ local.T @ (np.array(C.WRIST_CAM["eye"]) - np.asarray(tool_vec))
    return np.array([xy[0] - offset[0], xy[1] - offset[1], C.SURFACE_DZ + LOOK_TIP_CLEARANCE])


def camera_model_from_stage(stage, path, resolution):
    """Pose and intrinsics of a USD camera as a CameraModel (same recipe as capture_scene)."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    cam = UsdGeom.Camera(prim)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    return CameraModel(
        dict(
            resolution=list(resolution),
            focal_length_mm=float(cam.GetFocalLengthAttr().Get()),
            horizontal_aperture_mm=float(cam.GetHorizontalApertureAttr().Get()),
            vertical_aperture_mm=float(cam.GetVerticalApertureAttr().Get()),
            camera_to_world=[[float(xf[r][c]) for c in range(4)] for r in range(4)],
            table_top_z=float(C.TABLE_TOP_Z + C.SURFACE_DZ),
            robot_base_xy=[0.0, 0.0],
        )
    )


def noise_sigma(im):
    """Immerkaer's noise estimate on luminance; ~0 for a clean render."""
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    response = np.abs(cv2.filter2D(g, -1, kernel)[1:-1, 1:-1]).sum()
    return float(np.sqrt(np.pi / 2) * response / (6 * (w - 2) * (h - 2)))


def adaptive_denoise(im):
    """Denoise in proportion to measured noise; a clean image is returned untouched.

    Heavy denoising of a clean image smears faint edges and costs accuracy.
    """
    sigma = noise_sigma(im)
    if sigma < NOISE_FLOOR:
        return im
    h = float(np.clip(0.8 * sigma, 3, 30))
    return cv2.fastNlMeansDenoisingColored(im, None, h, h, 5, 15)


def normalise_wrist(im):
    """Adaptive denoise, then repaint the table neutral grey.

    The table is nearly the whole wrist view, so its colour is the whole-image
    median; pixels with that chroma are repainted grey, which neutralises tinted
    tables without touching the cube.
    """
    im = adaptive_denoise(im)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    L0, a0, b0 = np.median(lab.reshape(-1, 3), axis=0)
    table = np.hypot(lab[:, :, 1] - a0, lab[:, :, 2] - b0) < TABLE_CHROMA_TOL
    shade = np.clip(lab[:, :, 0] / max(L0, 1.0), 0.6, 1.15)
    out = im.astype(np.float32)
    out[table] = (TABLE_TARGET * shade[table])[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def refine_yaw(frames, overhead_yaw, xy):
    """Median wrist yaw over `frames` [(bgr image, CameraModel)], checked against overhead.

    Returns a dict with ok, reason, yaw (deg mod 90, when ok) and the per-frame yaws.
    Any frame without a measurable top face, a gross disagreement with the overhead
    yaw, or unstable frames rejects the pick rather than guessing.
    """
    yaws = [top_face_yaw(normalise_wrist(im), cam, xy) for im, cam in frames]
    result = dict(ok=False, reason=None, yaw=None, frame_yaws=yaws, overhead_yaw=float(overhead_yaw))
    if any(y is None for y in yaws):
        result["reason"] = "wrist_top_face_not_found"
        return result
    delta = np.array([square_yaw(y - overhead_yaw) for y in yaws])
    result["delta_deg"] = delta.tolist()
    if np.abs(delta).max() > WRIST_AGREEMENT_DEG:
        result["reason"] = "wrist_overhead_yaw_disagree"
    elif np.ptp(delta) > MAX_FRAME_SPREAD_DEG:
        result["reason"] = "wrist_frames_unstable"
    else:
        result.update(ok=True, yaw=float(square_yaw(overhead_yaw + np.median(delta))))
    return result
