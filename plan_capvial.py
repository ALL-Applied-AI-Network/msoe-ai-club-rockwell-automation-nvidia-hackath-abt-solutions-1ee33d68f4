"""Plan the cap/vial task: pick caps and vials, set them aside; leave the cubes alone.

A thin wrapper over plan_sort (unchanged): it points the sort task's destination slots
at the side areas in capvial_config and runs plan_sort's own planning, evaluation and
overlay. Narrow-end-up caps and tubes are picked; wide-end-up caps are skipped because
the gripper can only hold the plug (the executor rejects them even in an edited plan).

    .venv/bin/python plan_capvial.py --run runs/example [--image world_camera.png]
"""

import sys

import capvial_config as K
import plan_sort
import scene_config as C

# Read by plan_sort.build_plan at call time.
SLOT_OVERRIDES = {"SORT_CAP_TARGETS": list(K.CAP_SLOTS), "SORT_VIAL_TARGETS": list(K.VIAL_SLOTS)}


def main():
    for name, value in SLOT_OVERRIDES.items():
        setattr(C, name, value)
    argv = sys.argv[1:]
    if "--task" not in argv:
        argv += ["--task", "sort"]
    sys.argv = [sys.argv[0], *argv]
    plan_sort.main()


if __name__ == "__main__":
    main()
