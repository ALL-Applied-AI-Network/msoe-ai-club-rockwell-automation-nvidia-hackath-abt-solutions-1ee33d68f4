"""Evaluate detector weights on corruption types/strengths that augmentation did NOT cover.

    .venv-yolo/bin/python holdout_eval.py <eval_detector args...>
"""
import sys

import cv2
import numpy as np

import eval_detector
import perturb_eval as P


def gamma(g):
    return lambda im, rng: (255 * (im / 255.0) ** g).clip(0, 255).astype(np.uint8)


def colour_cast(b, g, r):
    return lambda im, rng: np.clip(im.astype(np.float32) * [b, g, r], 0, 255).astype(np.uint8)


def poisson(scale):
    return lambda im, rng: np.clip(rng.poisson(im / 255.0 * scale) / scale * 255, 0, 255).astype(np.uint8)


P.CASES.clear()
P.CASES.update({
    "dim_0.15": P.gain(0.15),
    "gamma_2.5 (dark midtones)": gamma(2.5),
    "jpeg_q8": P.jpeg(8),
    "noise_s45": P.noise(45),
    "poisson_low_light": poisson(20),
    "blue_cast+dim0.4": P.stacked(colour_cast(1.3, 0.9, 0.7), P.gain(0.4)),
    "table_near_black": P.table_tint((40, 40, 40)),
    "table_dark+dim0.4+jpeg15": P.stacked(P.table_tint((70, 70, 70)), P.gain(0.4), P.jpeg(15)),
    "blur_9+noise20+jpeg25": P.stacked(P.blur(9), P.noise(20), P.jpeg(25)),
})
eval_detector.CASES = P.CASES
eval_detector.main()
