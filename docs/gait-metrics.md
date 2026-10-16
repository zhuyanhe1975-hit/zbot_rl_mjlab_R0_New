# Gait metrics

`python gait_metrics.py <TASK> <checkpoint.pt> [--speed 0.25]` rolls out 16 environments for
1000 control steps (20 s) at a fixed forward command, drops the first 250 steps, and prints:

| Metric | Meaning | Ideal |
|---|---|---|
| `foot{0,1} pitch/roll mean, p90` | sole tilt (deg) while that foot is loaded, in the Zbot forward/left frame | small; p90 under about 2 for a full-sole landing |
| `double-support`, `flight` | fraction of steps with both feet down / both feet up | walking: small, 0 |
| `period`, `cv` | mean time between touchdowns of one foot (s) and its coefficient of variation | cv under about 0.05 |
| `alternation phase` | when foot 1 lands after foot 0, as a fraction of the period; std over strides | 0.50, std near 0 |
| `mirror error joint i/(7-i)` | RMS of q_i(t) + q_(7-i)(t + T/2), divided by std of q_i | 0 perfect, about 1.4 unrelated |

The sole normal is taken from the `foot_0_sole` / `foot_1_sole` sites. foot_1's site z-axis points out of
the foot; the tilt uses the absolute normal so this does not matter. Both soles are a flat
105 x 105 mm face (8621 mm^2), so a tilt of 5 degrees already reduces ground contact to an edge or corner.

## Reference: velocity-energy-tracking, 500 iterations, vx = 0.25 m/s

Checkpoint `bests/velocity_energy_tracking_2026-10-10_iter499`:

| Metric | Value |
|---|---|
| foot0 pitch / roll, mean (p90) | 3.4 (7.5) / 3.9 (6.8) deg |
| foot1 pitch / roll, mean (p90) | 3.2 (6.8) / 3.7 (6.5) deg |
| double-support / flight | 0.02 / 0.00 |
| period / cv | 1.08 s / 0.05 |
| alternation phase | 0.50, std 0.02 |
| mirror error joint1/6, 2/5, 3/4 | 0.68, 0.34, 0.29 |

## Caveats

- Metrics are only meaningful when the policy does not fall. The script does not check this itself:
  after a fall the environment resets inside the window and the numbers mix several episodes. Check the
  training log's mean episode length (should be near 1000) before trusting them.
- Contact is counted from the ground-contact sensor; a bouncing foot can register extra touchdowns, so
  absolute step frequency is approximate. Compare models with each other, not against a physical value.
