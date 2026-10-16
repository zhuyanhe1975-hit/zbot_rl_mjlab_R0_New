# RobStride RS00 actuator

Source: *RS00使用说明书* (260713), Lingzu Times (RobStride Dynamics). Page numbers below are the
PDF's own. Constants live in `src/zbot_rl_mjlab/robot.py`.

## What the manual says

| Item | Value | Page |
|---|---|---|
| Rated voltage / range | 48 V / 24-60 V | 6 |
| No-load speed | 315 rpm +/-10 % (about 33 rad/s), no-load current 0.5 Arms | 6 |
| Rated load | 5 N*m, measured at 100 rpm on a 90x85 mm aluminium heatsink | 6 |
| Peak load | 14 N*m (rated phase current 4.7 Apk, maximum 15.5 Apk) | 7 |
| Back-EMF / torque constant | 9.5 Vrms/krpm, 1.48 N*m/Arms | 7 |
| T-N curve at 48 V | 14 N*m at 100 rpm, about 12 at 150, 9.5 at 200, 6.5 at 250, 4 at 280, 2 at 300, 0.5 at 315 rpm (read off the figure, +/-0.5 N*m) | 7 |
| Rotating overload | 14 N*m for 3 s, 13 for 7 s, 12 for 10 s, 10 for 20 s, 8 for 53 s, 7 for 94 s, 6 for 285 s, 5 continuous | 8 |
| Stalled overload | 14 N*m for 1.5 s ... 5 N*m for 110 s, 3.6 N*m continuous | 9 |
| Gear ratio, mass | 10:1, 310 g +/-3 g | 9 |
| Encoder | 14 bit single-turn absolute | 10 |
| MIT ("运控") control law | `t_ref = Kd*(v_set - v) + Kp*(p_set - p) + t_ff`, summed into a reference current and tracked by a current loop | 41 |
| MIT command ranges | angle +/-12.57 rad, speed +/-33 rad/s, torque +/-14 N*m, **Kp 0-500, Kd 0-5** (16-bit) | 32 |

Not in the manual: rotor / reflected inertia, gearbox friction, backlash, current-loop bandwidth.

## What the simulation uses

`robot_cfg()` uses mjlab's `BuiltinPositionActuatorCfg`: an implicit-integration position servo,
`kp = 15 N*m/rad`, `kd = 1.5 N*m*s/rad`, `armature = 0.001`, **torque limit 20 N*m**. This is the
actuator every archived checkpoint was trained with, and it is kept so they keep replaying.

* The gains are inside the MIT range, and so is the deploy tasks' +/-30 % gain randomization.
* The simulator therefore allows 43 % more torque than the hardware peak (14 N*m) and has **no speed
  limit**: it does not apply the falling T-N curve. Real motors give 14 N*m only up to about 100 rpm
  (10.5 rad/s) and nothing at 315 rpm (33 rad/s). A gait that works in simulation may therefore ask
  for torque or speed the module cannot deliver.
* `actuator_usage.py` measures exactly that for a given checkpoint: per joint, torque peak / rms
  against the 14 and 5 N*m limits, joint speed against 100 and 315 rpm, the share of samples where
  the torque in the direction of motion exceeds the datasheet curve, and a thermal check against the
  rotating-overload table. Run it before trusting a gait on hardware.

## What was tried and did not work

Two datasheet-faithful actuators were tried and reverted:

1. `DcMotorActuatorCfg` (explicit Python PD, 14 N*m, fitted T-N line): a policy trained under the old
   actuator fell within about 20 steps at every speed (807-883 falls in 16 environments x 1000 steps).
   The explicit PD is evaluated once per 5 ms step, which the real servo (far faster) is not. A stability
   estimate suggests `kd*dt/I` may exceed 2, but the effective inertia was not measured.
2. `BuiltinPdActuatorCfg` (implicit PD with 12 control inputs) and `BuiltinDcMotorActuatorCfg`:
   `play.py` stopped with `size of tensor a (12) must match size of tensor b (6)`. The traceback was
   not obtained, so the cause is unknown; the 12-wide control vector is the likely trigger.

Both changes made the existing models impossible to replay. Do not switch the actuator again without
checking `play.py` on the archived checkpoints, and expect to retrain afterwards.

## Not modeled

* The falling T-N curve and the 14 N*m peak (see above), the +/-10 % spread on no-load speed.
* Reflected rotor inertia (`armature` is a guess), gearbox friction, backlash, current-loop dynamics,
  supply-voltage sag.
* Encoder quantization (14 bit is about 0.0004 rad) and the MIT feedback speed clip at +/-33 rad/s.
