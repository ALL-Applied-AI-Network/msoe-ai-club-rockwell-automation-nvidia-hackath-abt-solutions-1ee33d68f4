"""Where the cap/vial task sets objects down: aside, clear of the cube field.

The cubes lie in x 0.24-0.46 m, y -0.15..0.28 m (scene_config.hackathon_cubes) and are
left alone in this task. Caps go to the +y side, tubes to the -y side.
"""

import scene_config as C

# Starter's existing cap grid (IK-reachable); 8 slots on the +y side, beyond the cubes.
CAP_SLOTS = list(C.SORT_CAP_TARGETS)

# Tubes lie along world X (jaws close along Y) and are 140 mm long, so a slot's centre
# at x = 0.28 spans x 0.21-0.35. 60 mm pitch keeps a neighbouring tube clear of the fingers.
VIAL_SLOTS = [(0.28, -0.34), (0.28, -0.40)]

# Cubes go to the +y side, opposite the tubes: 2 rows x 3 columns at 60 mm pitch, about
# 5 cm beyond the cube field's top edge (IK-reachable; y > 0.40 runs out of reach).
CUBE_SLOTS = [(x, y) for y in (0.34, 0.40) for x in (0.24, 0.30, 0.36)]

# Footprint margins used to keep the slots clear of the cube field.
CUBE_FIELD_X = (0.24, 0.46)
CUBE_FIELD_Y = (-0.15, 0.28)
SLOT_CLEARANCE = 0.02
