"""Run a script with scene_config patched from a layout file; no existing file changes.

The scene is built from scene_config's object lists, so patching them before the
script imports the scene code lays the objects out as the layout says. Use it for
both the capture and the executor so they build the same scene.

    $ISAAC/python.sh with_layout.py layout_1.json capture_scene.py --out runs/x
    $ISAAC/python.sh with_layout.py layout_1.json sort_capvial.py --run runs/x
"""

import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: with_layout.py LAYOUT.json SCRIPT.py [script args...]")
    layout_path, script = os.path.abspath(sys.argv[1]), sys.argv[2]
    with open(layout_path) as f:
        layout = json.load(f)
    sys.path.insert(0, HERE)
    import scene_config as C

    C.SORT_CUBES = [tuple(c) for c in layout["cubes"]]
    C.SORT_CUBE_YAWS_DEG = list(layout["cube_yaws_deg"])
    C.SORT_VIALS = [tuple(v) for v in layout["vials"]]
    C.SORT_CAPS = [tuple(c) for c in layout["caps"]]
    C.CAP_ORIENTATIONS = list(layout["cap_orientations"])
    sys.argv = [script, *sys.argv[3:]]
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")


main()
