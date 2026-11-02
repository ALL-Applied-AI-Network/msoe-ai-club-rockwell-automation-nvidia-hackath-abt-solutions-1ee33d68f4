"""Host-side checks for the cap/vial task: detections, slots and planning.

Synthetic inputs only: no Isaac Sim, no detector weights.
"""

import unittest
from unittest.mock import patch

import numpy as np

import capvial_config as K
import capvial_detections as D
import fk
import ik
import plan_capvial
import scene_config as C
from plan_sort import build_plan
from test_support import camera_model as model


class DetectionTests(unittest.TestCase):
    def test_vial_axis_is_the_short_side_midpoints(self):
        poly = np.array([[100, 50], [200, 50], [200, 62], [100, 62]], float)  # 100 x 12 px
        ends = D.axis_endpoints(poly)
        np.testing.assert_allclose(sorted(ends.tolist()), [[100, 56], [200, 56]])

    def test_units_and_cap_orientation_classes(self):
        poly = [[96, 54], [144, 54], [144, 108], [96, 108]]
        narrow = D.detection(D.CAP_NARROW_UP, 0.9, poly, (1920, 1080))
        wide = D.detection(D.CAP_WIDE_UP, 0.8, poly, (1920, 1080))
        self.assertEqual((narrow["orientation"], wide["orientation"]), ("narrow_end_up", "wide_end_up"))
        np.testing.assert_allclose(narrow["center"], [120 / 1920, 81 / 1080])
        self.assertEqual(narrow["category"], "cap")

    def test_vial_detection_has_axis_and_confidence(self):
        d = D.detection(D.VIAL, 0.7, [[0, 0], [100, 0], [100, 10], [0, 10]], (1920, 1080))
        self.assertEqual((d["category"], len(d["axis_endpoints"])), ("vial", 2))
        self.assertAlmostEqual(d["axis_confidence"], 0.7)


class SlotTests(unittest.TestCase):
    def test_slots_are_clear_of_the_cube_field(self):
        pad = K.SLOT_CLEARANCE
        for _, y in K.CAP_SLOTS:
            self.assertGreaterEqual(y - C.CAP_RADIUS, K.CUBE_FIELD_Y[1] + pad)
        for _, y in K.VIAL_SLOTS:
            self.assertLessEqual(y + C.VIAL_RADIUS, K.CUBE_FIELD_Y[0] - pad)

    def test_vial_slots_keep_neighbours_clear_of_the_fingers(self):
        ys = sorted(y for _, y in K.VIAL_SLOTS)
        for a, b in zip(ys, ys[1:]):
            self.assertGreaterEqual(b - a, 0.05)

    def test_every_slot_is_reachable_with_the_tool_pointing_down(self):
        # Nominal tool length stands in for the rendered tip offset; the sim run is the
        # authoritative check.
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
        q0 = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q0, _ = ik.solve(q0, np.array([0.30, 0.0, 0.24]), [0, 0, -1], limits, tool_offset=tool)
        cases = [(xy, C.SURFACE_DZ + C.CAP_PICK_HEIGHT, None) for xy in K.CAP_SLOTS]
        cases += [(xy, C.SURFACE_DZ + C.TIP_CLEARANCE, (0, 1, 0)) for xy in K.VIAL_SLOTS]
        for (x, y), z, spread in cases:
            for lift in (0.0, 0.13):
                _, ok = ik.solve(q0, np.array([x, y, z + lift]), [0, 0, -1], limits,
                                 tool_offset=tool, spread_dir=spread, iters=400)
                self.assertTrue(ok, f"slot ({x}, {y}) lift {lift} unreachable")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.cam = model(C.TABLE_CAM)
        for patcher in (patch.object(C, "JAW_OPEN", 0.012), patch.multiple(C, **plan_capvial.SLOT_OVERRIDES)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def cap(self, xy, orientation, conf=0.95):
        uv = self.cam.project([*xy, self.cam.table_z + C.CAP_HEIGHT / 2]) / [self.cam.W, self.cam.H]
        return dict(category="cap", label="yellow cap", colour="yellow", center=uv.tolist(),
                    orientation=orientation, orientation_confidence=conf)

    def vial(self, centre, yaw_deg, conf=0.9):
        axis = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))])
        z = self.cam.table_z + C.VIAL_RADIUS + 0.001
        ends = [self.cam.project([*(np.asarray(centre) + s * C.VIAL_LENGTH / 2 * axis), z]) / [self.cam.W, self.cam.H]
                for s in (-1, 1)]
        return dict(category="vial", label="clear vial", colour="clear",
                    center=((ends[0] + ends[1]) / 2).tolist(), axis_endpoints=[e.tolist() for e in ends],
                    axis_confidence=conf)

    def test_picks_go_to_the_side_slots_and_wide_caps_stay(self):
        objects = [self.cap((0.40, 0.05), "narrow_end_up"), self.cap((0.34, -0.05), "wide_end_up"),
                   self.vial((0.40, -0.01), 10.0)]
        plan = build_plan({"coordinate_convention": "unit", "objects": objects}, self.cam, task="sort")
        picks = {p["category"]: p for p in plan["picks"]}
        self.assertEqual(set(picks), {"cap", "vial"})
        np.testing.assert_allclose(picks["cap"]["place_xy"], K.CAP_SLOTS[0])
        np.testing.assert_allclose(picks["vial"]["place_xy"], K.VIAL_SLOTS[0])
        self.assertIn("wide_end_up", [s["skip_reason"] for s in plan["skipped"]])

    def test_vial_jaws_close_perpendicular_to_the_tube(self):
        plan = build_plan({"coordinate_convention": "unit", "objects": [self.vial((0.40, -0.01), 35.0)]},
                          self.cam, task="sort")
        pick = plan["picks"][0]
        self.assertAlmostEqual(abs(np.dot(pick["axis_world"], pick["spread_dir"][:2])), 0.0, places=6)
        self.assertAlmostEqual(np.degrees(np.arctan2(pick["axis_world"][1], pick["axis_world"][0])) % 180, 35.0, delta=1.0)

    def test_no_cube_is_ever_picked(self):
        cube = dict(category="cube", label="red cube", colour="red", center=[0.5, 0.5])
        plan = build_plan({"coordinate_convention": "unit", "objects": [cube, self.cap((0.4, 0.05), "narrow_end_up")]},
                          self.cam, task="sort")
        self.assertEqual([p["category"] for p in plan["picks"]], ["cap"])

    def test_excess_vials_beyond_the_slots_are_skipped_not_misplaced(self):
        objects = [self.vial((0.40, y), 10.0) for y in (-0.02, 0.10, 0.20)]
        plan = build_plan({"coordinate_convention": "unit", "objects": objects}, self.cam, task="sort")
        self.assertEqual(len(plan["picks"]), len(K.VIAL_SLOTS))
        self.assertIn("no_vial_destination", [s["skip_reason"] for s in plan["skipped"]])


if __name__ == "__main__":
    unittest.main()
