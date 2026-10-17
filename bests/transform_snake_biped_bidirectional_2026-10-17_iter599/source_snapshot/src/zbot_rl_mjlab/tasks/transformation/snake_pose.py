"""Horizontal zero-angle reset pose for assets/zbot_6/ZBOT6.xml.

The root rolls -90 degrees so the chain runs along world +Y. Its 52.5 mm
collision radius sets the ground clearance; the chain is centered about Y=0.
This is an initial configuration, not a reference motion.

Verified with gravity and the task's ground friction and 15/1.5 position servos:
after two seconds, COM height is 52.45 mm, max module COM height is 52.82 mm,
root speed is 0.26 mm/s, and maximum contact penetration is below 0.04 mm.
"""

SNAKE_ROOT_POS = (0.0, -0.33, 0.0525)
SNAKE_ROOT_QUAT = (0.7071067811865476, -0.7071067811865476, 0.0, 0.0)
SNAKE_JOINT_POS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
