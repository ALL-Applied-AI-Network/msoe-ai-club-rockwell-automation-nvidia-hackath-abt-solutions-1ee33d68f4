"""Host-side checks for the random cap/vial layouts."""

import unittest

import numpy as np

import capvial_config as K
import capvial_layouts as L
import scene_config as C

SEEDS = range(60)


def objects(layout):
    out = [("vial", L.footprint("vial", x, y, yaw)) for x, y, yaw in layout["vials"]]
    out += [("cap", L.footprint("cap", x, y)) for x, y in layout["caps"]]
    out += [("cube", L.footprint("cube", x, y)) for x, y, _ in layout["cubes"]]
    return out


class LayoutTests(unittest.TestCase):
    def test_counts_and_default_cube_colour_order(self):
        for seed in (0, 1, 2):
            lay = L.sample(seed)
            self.assertEqual((len(lay["cubes"]), len(lay["vials"]), len(lay["caps"])), (6, 2, 2))
            self.assertEqual([c[2] for c in lay["cubes"]], list(L.CUBE_COLOURS))
            self.assertEqual(len(lay["cube_yaws_deg"]), 6)
            self.assertEqual(len(lay["cap_orientations"]), 2)

    def test_same_seed_same_layout_different_seed_different_layout(self):
        self.assertEqual(L.sample(7), L.sample(7))
        self.assertNotEqual(L.sample(7), L.sample(8))

    def test_every_pair_keeps_its_minimum_gap(self):
        for seed in SEEDS:
            objs = objects(L.sample(seed))
            for i, (ka, a) in enumerate(objs):
                for kb, b in objs[i + 1 :]:
                    self.assertGreaterEqual(L.gap(a, b), L.min_gap(ka, kb) - 1e-9, f"seed {seed}: {ka} vs {kb}")

    def test_objects_stay_in_their_zones_and_clear_of_the_set_down_slots(self):
        for seed in SEEDS:
            lay = L.sample(seed)
            for x, y, yaw in lay["vials"]:
                fp = L.footprint("vial", x, y, yaw)
                self.assertGreater(fp[:, 1].min(), K.VIAL_SLOTS[0][1] + 0.05, f"seed {seed}")
                self.assertGreater(fp[:, 0].min(), 0.2, f"seed {seed}")  # reach
            for x, y in lay["caps"]:
                self.assertLess(y + C.CAP_RADIUS, min(s[1] for s in K.CAP_SLOTS) - 0.02)
            for x, y, _ in lay["cubes"]:
                self.assertTrue(L.CUBE_X[0] <= x <= L.CUBE_X[1] and L.CUBE_Y[0] <= y <= L.CUBE_Y[1])

    def test_both_cap_orientations_occur(self):
        seen = {o for seed in SEEDS for o in L.sample(seed)["cap_orientations"]}
        self.assertEqual(seen, {"narrow_end_up", "wide_end_up"})


if __name__ == "__main__":
    unittest.main()
