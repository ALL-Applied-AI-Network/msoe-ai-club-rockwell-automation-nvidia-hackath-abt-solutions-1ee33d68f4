"""Cube detection without a VLM: colour segmentation plus calibrated silhouette fit.

Drop-in alternative to `perceive.py --stack-cubes`. Writes detections.json from the
captured table image only; ground truth is never read. Pose confidence is the
silhouette IoU of the image fit, so the planner's confidence and fit gates stay
meaningful. Runs under the host interpreter.

    .venv/bin/python perceive_opencv.py --run runs/example

The image is first normalised (denoised, table colour and exposure canonicalised)
and the planner is expected to refine on that same image (`plan_sort.py --image
world_camera_norm.png`).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import scene_config as C
from backproject import CameraModel
from cube_pose import refine_pose, silhouette, square_yaw

# Same hue gates as cube_pose.refine_pose so detection and pose fit agree.
COLOUR_GATES = {
    "yellow": lambda h: (h >= 12) & (h <= 42),
    "blue": lambda h: (h >= 95) & (h <= 135),
    "red": lambda h: (h <= 12) | (h >= 170),
}
AREA_RATIO = (0.5, 1.6)  # blob area / expected projected cube area
MIN_IOU = 0.85

# Calibrated baseline render: neutral table level, and median cube-top colour (BGR).
TABLE_TARGET = 210.0
CUBE_REF_BGR = {
    "yellow": (110, 176, 187),
    "blue": (163, 93, 53),
    "red": (71, 77, 179),
}
TABLE_CHROMA_TOL = 10.0  # Lab a/b distance treated as "table"


def _work_region(im, cam):
    H, W = im.shape[:2]
    x0, x1 = C.BACKDROP_X
    w = C.BACKDROP_WIDTH / 2
    corners = [cam.project([x, y, cam.table_z]) for x, y in ((x0, -w), (x1, -w), (x1, w), (x0, w))]
    region = np.zeros((H, W), np.uint8)
    cv2.fillPoly(region, [np.round(corners).astype(np.int32)], 1)
    return region


def candidate_blobs(im, cam):
    """Colour blobs with a plausible cube size on the work surface (no pose fit)."""
    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    blobs, rejected = [], []
    for colour, gate in COLOUR_GATES.items():
        mask = (gate(hsv[:, :, 0]) & (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 35)).astype(
            np.uint8
        )
        n, labels, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = int(stats[i, 4])
            if area < 40:
                continue
            u, v = centres[i]
            xy = cam.to_plane(u, v, z=cam.table_z + C.CUBE_EDGE / 2)
            note = dict(colour=colour, centre_px=[round(float(u)), round(float(v))], area=area)
            if xy is None or not (
                C.BACKDROP_X[0] <= xy[0] <= C.BACKDROP_X[1]
                and abs(xy[1]) <= C.BACKDROP_WIDTH / 2
            ):
                rejected.append(dict(note, reason="off_work_surface"))
                continue
            expected = float((silhouette(cam, xy[:2], 0.0, (H, W)) > 0).sum())
            ratio = area / max(expected, 1.0)
            if not AREA_RATIO[0] <= ratio <= AREA_RATIO[1]:
                rejected.append(dict(note, reason="size_not_cube", area_ratio=round(ratio, 2)))
                continue
            blobs.append(dict(colour=colour, centre=(float(u), float(v)), xy=xy[:2],
                              stats=stats[i], mask=labels == i))
    return blobs, rejected


def normalise(im, cam):
    """Return a canonical-looking image: denoised, table neutral grey, cubes at known exposure.

    Two separate physical effects are handled separately. A table whose colour
    changes (cubes unchanged) is repainted to neutral grey by chroma. A change in
    lighting (everything scaled) is undone per cube blob, against the calibrated
    baseline cube colours. Quality gates downstream are not touched.
    """
    im = cv2.fastNlMeansDenoisingColored(im, None, 7, 7, 5, 15)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    region = _work_region(im, cam) > 0
    L0, a0, b0 = np.median(lab[region], axis=0)
    table = np.hypot(lab[:, :, 1] - a0, lab[:, :, 2] - b0) < TABLE_CHROMA_TOL
    shade = np.clip(lab[:, :, 0] / max(L0, 1.0), 0.6, 1.15)
    out = im.astype(np.float32)
    out[table] = (TABLE_TARGET * shade[table])[:, None]

    blobs, _ = candidate_blobs(np.clip(out, 0, 255).astype(np.uint8), cam)
    for b in blobs:
        ref = np.mean(CUBE_REF_BGR[b["colour"]])
        seen = np.mean(np.median(im[b["mask"]], axis=0))
        gain = ref / max(seen, 1.0)
        near = cv2.dilate(b["mask"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        sel = near & ~table
        out[sel] = out[sel] * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def _square_mask(n, centre, side_px, yaw_deg, supersample=4):
    t = np.deg2rad(yaw_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = (np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) * side_px / 2) @ rot.T + centre
    m = np.zeros((n * supersample, n * supersample), np.uint8)
    cv2.fillPoly(m, [np.round(pts * supersample).astype(np.int32)], 1)
    return cv2.resize(m.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA)


def top_face_yaw(im, cam, xy, half=0.012, scale=0.00025):
    """Yaw (deg, mod 90) from the cube's top face, or None if it can't be isolated.

    The top-face plane is rectified to a top-down view (0.25 mm/px). The top face
    is the brightest part of the cube there, so a 10 mm square template is fitted
    to it. This beats the whole-cube silhouette because side faces don't enter.
    """
    z_top = cam.table_z + C.CUBE_EDGE
    n = int(round(2 * half / scale))
    src = np.array(
        [cam.project([xy[0] + sx * half, xy[1] + sy * half, z_top]) for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))],
        np.float32,
    )
    dst = np.array([(0, 0), (n, 0), (n, n), (0, n)], np.float32)  # rows run toward -y
    top = cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (n, n), flags=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    cube = hsv[:, :, 1] > 45
    if cube.sum() < 200:
        return None
    v = hsv[:, :, 2]
    level, _ = cv2.threshold(v[cube].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    face = cv2.morphologyEx((cube & (v > level)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(face, 8)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, 4]))
    mask = cv2.GaussianBlur((labels == biggest).astype(np.float32), (0, 0), 1.0)
    side_px = C.CUBE_EDGE / scale

    def score(deg):
        return float((mask * _square_mask(n, centres[biggest], side_px, deg)).sum())

    best = max(np.arange(-45, 45, 0.5), key=score)
    s0, s1, s2 = score(best - 0.5), score(best), score(best + 0.5)
    curve = s0 - 2 * s1 + s2
    deg = best + (0.25 * (s0 - s2) / curve if abs(curve) > 1e-9 else 0.0)
    return square_yaw(-deg)  # template is drawn in image rows, which run toward -y


def find_cubes(im, cam):
    H, W = im.shape[:2]
    blobs, rejected = candidate_blobs(im, cam)
    objects = []
    for b in blobs:
        colour, (u, v) = b["colour"], b["centre"]
        xy, yaw, iou = refine_pose(
            im, cam, {"colour": colour}, b["xy"], 0.0, search_half_width=45
        )
        top_yaw = top_face_yaw(im, cam, xy)
        if top_yaw is not None:
            yaw = top_yaw  # seeds the planner's edge pose; planner re-checks both
        x, y, w, h = (int(b["stats"][k]) for k in range(4))
        t = np.deg2rad(yaw)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        edge = [
            cam.project(
                [*(xy + rot @ np.array([a, -1]) * C.CUBE_EDGE / 2), cam.table_z + C.CUBE_EDGE]
            )
            for a in (-1, 1)
        ]
        objects.append(
            dict(
                label=f"{colour} cube",
                category="cube",
                colour=colour,
                center=[u / W, v / H],
                bbox=[x / W, y / H, (x + w) / W, (y + h) / H],
                graspable=True,
                localisation_source="opencv_colour_blob",
                pose_source="calibrated_image_silhouette_fit",
                silhouette_iou=round(float(iou), 4),
                # Fit quality only; below MIN_IOU, or with no top-face yaw, skip it.
                pose_confidence=(
                    round(float(iou), 4) if iou >= MIN_IOU and top_yaw is not None else 0.0
                ),
                top_face_yaw_deg=None if top_yaw is None else round(float(top_yaw), 3),
                top_edge_endpoints=(np.array(edge) / [W, H]).tolist(),
                image_yaw_deg=round(square_yaw(yaw), 3),
            )
        )
    return objects, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    run = Path(ap.parse_args().run)
    im = cv2.imread(str(run / "world_camera.png"))
    if im is None:
        raise SystemExit(f"Missing world_camera.png in {run}; run capture_scene.py first")
    cam = CameraModel(str(run / "camera.json"))
    # The planner must refine on the same normalised image the detector used.
    im = normalise(im, cam)
    cv2.imwrite(str(run / "world_camera_norm.png"), im)
    objects, rejected = find_cubes(im, cam)
    det = {
        "source": "OpenCV",
        "coordinate_convention": "unit",
        "objects": objects,
        "rejected_blobs": rejected,
    }
    (run / "detections.json").write_text(json.dumps(det, indent=2) + "\n")
    counts = {c: sum(o["colour"] == c for o in objects) for c in COLOUR_GATES}
    print(f"detected cubes: {counts}; rejected blobs: {len(rejected)}")
    for r in rejected:
        print("  rejected:", r)


if __name__ == "__main__":
    main()
