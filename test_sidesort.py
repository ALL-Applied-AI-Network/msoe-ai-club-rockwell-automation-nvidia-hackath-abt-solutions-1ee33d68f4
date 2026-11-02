"""Host-side checks for the clear-the-centre planner (tubes -y, cubes +y, caps stay)."""

import unittest

import numpy as np

import capvial_config as K
import ik
import plan_sidesort as P
import scene_config as C
from test_support import camera_model as model


class SlotTests(unittest.TestCase):
    def test_cube_slots_are_on_the_opposite_side_clear_of_the_field(self):
        for x, y in K.CUBE_SLOTS:
            self.assertGreater(y - C.CUBE_EDGE, K.CUBE_FIELD_Y[1] + K.SLOT_CLEARANCE)
        self.assertEqual(len(K.CUBE_SLOTS), len(set(K.CUBE_SLOTS)))
        self.assertLess(max(y for _, y in K.VIAL_SLOTS), K.CUBE_FIELD_Y[0])  # tubes: other side

    def test_slots_keep_neighbours_clear_of_the_fingers(self):
        pts = np.array(K.CUBE_SLOTS)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                self.assertGreaterEqual(np.linalg.norm(pts[i] - pts[j]), 0.05)

    def test_every_cube_slot_is_reachable(self):
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)  # nominal; the sim run is authoritative
        q0 = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        q0, _ = ik.solve(q0, np.array([0.30, 0.0, 0.24]), [0, 0, -1], limits, tool_offset=tool)
        z = C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE
        for x, y in K.CUBE_SLOTS:
            for lift in (0.0, 0.13):
                _, ok = ik.solve(q0, np.array([x, y, z + lift]), [0, 0, -1], limits, tool_offset=tool,
                                 spread_dir=(0, 1, 0), iters=400)
                self.assertTrue(ok, f"cube slot ({x}, {y}) lift {lift} unreachable")


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.cam = model(C.TABLE_CAM)

    def uv(self, xy, z):
        return (self.cam.project([*xy, self.cam.table_z + z]) / [self.cam.W, self.cam.H]).tolist()

    def cube(self, xy, colour, confidence=0.95):
        edge = [
            (self.cam.project([xy[0] + dx, xy[1] - C.CUBE_EDGE / 2, self.cam.table_z + C.CUBE_EDGE])
             / [self.cam.W, self.cam.H]).tolist()
            for dx in (-C.CUBE_EDGE / 2, C.CUBE_EDGE / 2)
        ]
        return dict(category="cube", label=f"{colour} cube", colour=colour, center=self.uv(xy, C.CUBE_EDGE / 2),
                    top_edge_endpoints=edge, pose_confidence=confidence)

    def cap(self, xy):
        return dict(category="cap", label="yellow cap", colour="yellow", center=self.uv(xy, C.CAP_HEIGHT / 2),
                    orientation="narrow_end_up", orientation_confidence=0.95)

    def vial(self, centre, yaw):
        axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        z = self.cam.table_z + C.VIAL_RADIUS + 0.001
        ends = [(self.cam.project([*(np.asarray(centre) + s * C.VIAL_LENGTH / 2 * axis), z]) / [self.cam.W, self.cam.H])
                for s in (-1, 1)]
        return dict(category="vial", label="clear vial", colour="clear", center=((ends[0] + ends[1]) / 2).tolist(),
                    axis_endpoints=[e.tolist() for e in ends], axis_confidence=0.9)

    def plan(self, cubes, extra=()):
        det = {"coordinate_convention": "unit", "objects": [*cubes, *extra]}
        return P.build(det, self.cam, None, None)

    def six_cubes(self):
        pos = [(0.26, 0.20), (0.34, 0.22), (0.42, 0.18), (0.30, 0.05), (0.38, 0.0), (0.44, 0.10)]
        return [self.cube(p, c) for p, c in zip(pos, ("yellow", "yellow", "red", "red", "blue", "blue"))]

    def test_tubes_and_cubes_each_get_their_own_side_slot_and_caps_stay(self):
        extra = [self.vial((0.40, -0.05), 20.0), self.vial((0.34, -0.15), -30.0),
                 self.cap((0.30, -0.10)), self.cap((0.45, 0.05))]
        plan = self.plan(self.six_cubes(), extra)
        by = lambda k: [p for p in plan["picks"] if p["category"] == k]
        self.assertEqual((len(by("vial")), len(by("cube")), len(by("cap"))), (2, 6, 0))
        self.assertEqual({tuple(p["place_xy"]) for p in by("cube")}, set(K.CUBE_SLOTS))
        self.assertEqual({tuple(p["place_xy"]) for p in by("vial")}, set(K.VIAL_SLOTS))
        self.assertEqual(plan["task"], "sort")
        self.assertTrue(all("stack_level" not in p and "side_slot" in p for p in by("cube")))
        self.assertTrue(all(p["place_z"] == C.SURFACE_DZ + C.CUBE_TIP_CLEARANCE for p in by("cube")))

    def test_cube_pose_gates_still_apply_per_cube(self):
        cubes = self.six_cubes()
        cubes[2] = self.cube((0.42, 0.18), "red", confidence=0.4)
        plan = self.plan(cubes)
        picked = [p for p in plan["picks"] if p["category"] == "cube"]
        self.assertEqual(len(picked), 5)
        self.assertIn("cube_pose_low_confidence", [s["skip_reason"] for s in plan["skipped"]])

    def test_an_incomplete_colour_set_does_not_stop_the_sort(self):
        plan = self.plan(self.six_cubes()[:5])  # not 2/2/2: no tower, but cubes still move aside
        self.assertEqual(len([p for p in plan["picks"] if p["category"] == "cube"]), 5)

    def test_more_cubes_than_slots_are_skipped_not_misplaced(self):
        pos = [(0.26 + 0.03 * i, -0.10 + 0.06 * (i % 4)) for i in range(8)]
        plan = self.plan([self.cube(p, "red") for p in pos])
        self.assertEqual(len([p for p in plan["picks"] if p["category"] == "cube"]), len(K.CUBE_SLOTS))
        self.assertIn("no_cube_destination", [s["skip_reason"] for s in plan["skipped"]])

    def test_nearest_to_the_cube_side_go_first(self):
        plan = self.plan(self.six_cubes())
        ys = [p["xy"][1] for p in plan["picks"] if p["category"] == "cube"]
        self.assertEqual(ys, sorted(ys, reverse=True))


if __name__ == "__main__":
    unittest.main()
