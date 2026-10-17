# Snake-to-biped Beta checkpoint 2550

Replay with the explicit native Beta task:

```bash
./scripts/python.sh play.py Mjlab-Zbot-6dof-Transform-Snake-To-Biped-Beta --checkpoint-file bests/transform_snake_to_biped_beta_2026-10-17_iter2550/model_2550.pt
./bests/test.sh transform_snake_to_biped_beta_2026-10-17_iter2550
```

The supplied 4-environment, 16-second numerical evaluation reports 4/4 strict successes from the original zero-joint snake start. The original training used manual reward hot updates. The new production task reproduces the Beta architecture and adds an automatically latched preparation course; a complete cold-start training of that automation has not been verified.

`env.yaml` and `agent.yaml` preserve the actual run startup configuration. `final_effective_weights.json` preserves the weights and stage saved in checkpoint 2550. They deliberately describe different times. Do not load this checkpoint using the Gaussian task.

The native production Beta task also passed 4/4 strict numerical evaluations; see `native_evaluation.json` and `goal_audit.json`. No 60-second/16-environment test, ONNX/JIT export, or hardware validation is included.
