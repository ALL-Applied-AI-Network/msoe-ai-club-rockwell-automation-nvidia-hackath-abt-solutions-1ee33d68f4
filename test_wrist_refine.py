"""Host-side checks for the wrist-camera yaw refinement and its executor plumbing.

Synthetic inputs only: no Isaac Sim, no rendering.
"""

import unittest
from unittest.mock import patch

import numpy as np

import fk
import scene_config as C
import wrist_refine as W
from sort_scene import align_signs, segments


class LookGeometryTests(unittest.TestCase):
    def test_camera_not_tool_is_centred_on_the_cube(self):
        tool = np.array(C.TOOL_AXIS) * 0.2913
        tip = W.look_target((0.42, 0.18), tool)
        self.assertAlmostEqual(tip[2], C.SURFACE_DZ + W.LOOK_TIP_CLEARANCE)
        # The lens is ~97.9 mm off the tool axis, so the tip must sit that far
        # from the cube in the horizontal plane.
        eye = np.array(C.WRIST_CAM["eye"])
        self.assertAlmostEqual(np.hypot(*(tip[:2] - [0.42, 0.18])), np.hypot(eye[1], eye[2]), 3)

    def test_tip_clears_the_tallest_table_object(self):
        self.assertGreater(W.LOOK_TIP_CLEARANCE, C.CAP_HEIGHT + 0.01)


class RefineYawTests(unittest.TestCase):
    def refine(self, yaws, overhead):
        frames = [(None, None)] * len(yaws)
        with patch.object(W, "normalise_wrist", lambda im: im), patch.object(
            W, "top_face_yaw", side_effect=yaws
        ):
            return W.refine_yaw(frames, overhead, (0.4, 0.0))

    def test_agreeing_frames_give_the_median_wrist_yaw(self):
        r = self.refine([27.0, 27.4, 26.8], 30.0)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["yaw"], 27.0, places=6)

    def test_gross_disagreement_with_overhead_is_rejected(self):
        r = self.refine([45.0, 45.0, 45.0], 30.0)
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "wrist_overhead_yaw_disagree")

    def test_missing_top_face_is_rejected(self):
        r = self.refine([27.0, None, 27.0], 30.0)
        self.assertEqual(r["reason"], "wrist_top_face_not_found")

    def test_unstable_frames_are_rejected(self):
        r = self.refine([30.0, 34.5, 30.0], 30.0)
        self.assertEqual(r["reason"], "wrist_frames_unstable")

    def test_square_symmetry_wraps_at_45_degrees(self):
        r = self.refine([-44.0, -44.0, -44.0], 44.0)  # 88 deg apart == 2 deg mod 90
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["yaw"], -44.0, places=6)


class NormaliseWristTests(unittest.TestCase):
    def scene(self, table_bgr):
        im = np.empty((120, 160, 3), np.uint8)
        im[:] = table_bgr
        im[50:70, 70:90] = (40, 170, 200)  # saturated cube-like patch
        return im

    def test_tinted_table_becomes_neutral_and_cube_is_untouched(self):
        out = W.normalise_wrist(self.scene((235, 200, 160)))
        self.assertLess(int(out[5, 5].max()) - int(out[5, 5].min()), 6)
        self.assertAlmostEqual(float(out[5, 5].mean()), 210, delta=15)
        np.testing.assert_array_equal(out[60, 80], (40, 170, 200))

    def test_noise_is_measured_and_a_clean_image_reads_clean(self):
        clean = np.full((120, 160, 3), 200, np.uint8)
        rng = np.random.default_rng(0)
        noisy = np.clip(clean + rng.normal(0, 10, clean.shape), 0, 255).astype(np.uint8)
        self.assertLess(W.noise_sigma(clean), W.NOISE_FLOOR)
        self.assertGreater(W.noise_sigma(noisy), W.NOISE_FLOOR)


class ExecutorPlumbingTests(unittest.TestCase):
    pick = dict(
        category="cube",
        xy=[0.4, 0.0],
        pick_z=0.0,
        place_xy=list(C.CUBE_STACK_XY),
        place_z=0.0,
        spread_dir=[0.0, 1.0, 0.0],
        place_spread_dir=[0.0, 1.0, 0.0],
    )

    def test_look_segments_precede_and_feed_the_approach(self):
        home = np.array([0.3, 0.0, 0.24])
        look_tip = np.array([0.3, 0.0, 0.04])
        segs = segments(self.pick, home, look=(look_tip, np.array(W.LOOK_SPREAD)))
        names = [s[0] for s in segs]
        self.assertEqual(names[:3], ["look", "look_dwell", "approach"])
        np.testing.assert_allclose(segs[2][1], look_tip)  # approach leaves the look pose
        np.testing.assert_allclose(segs[-1][2], home)  # and the arm still returns home
        # Rebuilding after the look must not repeat it.
        again = segments(self.pick, home, start=look_tip)
        self.assertEqual(again[0][0], "approach")

    def test_without_a_look_the_sequence_is_unchanged(self):
        home = np.array([0.3, 0.0, 0.24])
        segs = segments(self.pick, home)
        self.assertEqual(segs[0][0], "approach")
        np.testing.assert_allclose(segs[0][1], home)

    def test_jaw_direction_sign_follows_the_current_wrist(self):
        q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        r, _ = fk.fk(q)
        current = r @ np.array(C.FINGER_SPREAD_AXIS)
        flipped = dict(self.pick, spread_dir=(-current).tolist())
        aligned = align_signs(flipped, q)
        self.assertGreater(np.dot(current, aligned["spread_dir"]), 0)
        self.assertGreater(
            np.dot(aligned["spread_dir"], aligned["place_spread_dir"]), 0
        )


if __name__ == "__main__":
    unittest.main()
