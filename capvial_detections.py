"""Turn cap/vial detector output (oriented boxes) into planner detections.

Pure numpy, so it is usable from any interpreter and testable on the host.
"""

import numpy as np

CAP_NARROW_UP, CAP_WIDE_UP, VIAL = 0, 1, 2
CLASS_NAMES = ["cap_narrow_up", "cap_wide_up", "vial"]


def axis_endpoints(poly):
    """Midpoints of an oriented box's two short sides: the ends of a tube's axis."""
    poly = np.asarray(poly, float)
    edges = [(poly[i], poly[(i + 1) % 4]) for i in range(4)]
    lengths = [np.linalg.norm(b - a) for a, b in edges]
    short = sorted(range(4), key=lambda i: lengths[i])[:2]
    return np.array([(edges[i][0] + edges[i][1]) / 2 for i in short])


def detection(cls, conf, poly_px, size):
    """One planner detection (unit image coordinates) from a detector oriented box."""
    width, height = size
    poly = np.asarray(poly_px, float) / [width, height]
    lo, hi = poly.min(0), poly.max(0)
    base = dict(
        center=poly.mean(0).tolist(),
        bbox=[float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])],
        graspable=True,
        detector_confidence=float(conf),
    )
    if cls == VIAL:
        return dict(
            base,
            label="clear vial",
            category="vial",
            colour="clear",
            axis_endpoints=axis_endpoints(poly).tolist(),
            axis_confidence=float(conf),
        )
    return dict(
        base,
        label="yellow cap",
        category="cap",
        colour="yellow",
        orientation="narrow_end_up" if cls == CAP_NARROW_UP else "wide_end_up",
        orientation_confidence=float(conf),
    )
