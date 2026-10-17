"""Low arc on the standing target's joint-angle branch.

Screened using native meshes, gravity, 15/1.5 PD and 5 Nm torque clipping:
8 s holding with 0 and +/-1 N lateral pulses for 0.2 s, about 5.1 mm loaded
support margin and 8.1 mm CoM-to-sole-segment offset after settling. This does
not certify a path to standing or a
transition from the original straight snake. See outputs/autonomous/
prebend_shallow_reset_validation.json for the screening conditions and traces.
"""

ARC_JOINT_POS = (0.0, 0.04, -0.04, 0.04, -0.04, 0.0)
ARC_ROOT_POS = (0.0, -0.33, 0.052594859168749175)
ARC_ROOT_QUAT = (
    0.5718926808530292,
    -0.42390736758012,
    0.5659527229498161,
    0.415859135626136,
)
