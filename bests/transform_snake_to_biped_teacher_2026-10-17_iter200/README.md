# Random-roll snake-to-biped teacher: iteration 200

Task: `Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher`.
Source run: `logs/rsl_rl/zbot_transform_snake_to_biped_teacher/2026-10-17_17-24-59`.
The archived model is the latest saved checkpoint (200); the training log continues to iteration 215. This archive preserves the saved model, not unsaved final weights.

Independent deterministic evaluation: 16 environments, 16 s, seed 42, random longitudinal roll ±90°, zero initial joint angles, no external pushes. Strict goal success 16/16; all episodes ended by timeout. Final sole tilt ≤0.069°, max joint error ≤0.037 rad, continuous final standing hold ≥13.78 s. Full traces are in evaluation.json; compact results in evaluation_summary.json. This is simulation evidence, not hardware or disturbance validation.

Actor36/Critic44, 3 mirrored Beta actions, maximum commanded target rate 2π rad/s. Five rewards: pose12, height1, first-rise velocity8, sole orientation1, failure−10. The velocity term permanently stops for the episode after base peak height reaches 0.23 m. No extra reward stages. Startup configuration is in env.yaml/agent.yaml; checkpoint reward metadata in final_effective_weights.json. Training events are saved separately.

Run from the repository root:

```bash
python play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta-Robust-Teacher \
  --checkpoint-file bests/transform_snake_to_biped_teacher_2026-10-17_iter200/model_200.pt \
  --env.scene.num-envs 4
```

Play samples a fresh seed per launch. Add `--env.seed 42` for reproducibility. Each reset samples the snake-axis roll independently. Initialization writes the actual root quaternion; the snake's longitudinal axis remains horizontal.

Source and robot assets are preserved in source_snapshot/ to protect against future recipe/schema changes. External MJLab and Python environment are not bundled; versions and Git provenance are in metadata.json. SHA256SUMS covers all files except itself. Do not restore only reward weights if future action/observation/physics definitions have changed.
