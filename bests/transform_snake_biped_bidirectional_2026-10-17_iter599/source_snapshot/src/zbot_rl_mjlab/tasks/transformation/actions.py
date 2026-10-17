"""Six independent policy actions control bounded joint target velocities."""

import math
from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


def bounded_target(current_target, action, limits, dt, max_speed=1.0):
    rate = torch.nan_to_num(action).clamp(-1.0, 1.0) * max_speed
    return (current_target + rate * dt).clamp(min=limits[..., 0], max=limits[..., 1])


@dataclass(kw_only=True)
class JointTargetRateActionCfg(JointPositionActionCfg):
    max_target_speed: float = 1.0
    target_limit: float | None = None
    joint_target_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def build(self, env):
        return JointTargetRateAction(self, env)


class JointTargetRateAction(JointPositionAction):
    def __init__(self, cfg: JointTargetRateActionCfg, env):
        super().__init__(cfg, env)
        self.rate_cfg = cfg
        self._named_target_bounds = None
        if cfg.joint_target_bounds:
            self._initialize_target_bounds()
        self._processed_actions.copy_(
            self._entity.data.default_joint_pos[:, self._target_ids]
        )
        if self._named_target_bounds is not None:
            self._processed_actions.copy_(self._clamp_target(self._processed_actions))

    def _initialize_target_bounds(self):
        bounds = self.rate_cfg.joint_target_bounds
        missing = bounds.keys() - set(self._target_names)
        if missing:
            raise ValueError(f"Unknown commanded target joints: {sorted(missing)}")
        named_bounds = self._processed_actions.new_tensor(
            [[-math.inf, math.inf]] * len(self._target_names)
        )
        for name, (low, high) in bounds.items():
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError(
                    f"Invalid commanded target bounds for {name}: {(low, high)}"
                )
            named_bounds[self._target_names.index(name)] = named_bounds.new_tensor(
                [low, high]
            )
        self._named_target_bounds = named_bounds
        limits = self._target_limits()
        if (limits[..., 0] > limits[..., 1]).any():
            raise ValueError(
                "Commanded target bounds do not intersect joint/target limits"
            )

    def _target_limits(self):
        limits = self._entity.data.joint_pos_limits[:, self._target_ids]
        named_bounds = getattr(self, "_named_target_bounds", None)
        if self.rate_cfg.target_limit is not None:
            if named_bounds is None:
                limits = limits.clamp(
                    -self.rate_cfg.target_limit, self.rate_cfg.target_limit
                )
            else:
                limits = torch.stack(
                    (
                        limits[..., 0].clamp(min=-self.rate_cfg.target_limit),
                        limits[..., 1].clamp(max=self.rate_cfg.target_limit),
                    ),
                    dim=-1,
                )
        if named_bounds is not None:
            limits = torch.stack(
                (
                    torch.maximum(limits[..., 0], named_bounds[:, 0]),
                    torch.minimum(limits[..., 1], named_bounds[:, 1]),
                ),
                dim=-1,
            )
        return limits

    def _clamp_target(self, target, env_ids: slice | torch.Tensor = slice(None)):
        limits = self._target_limits()[env_ids]
        return target.clamp(min=limits[..., 0], max=limits[..., 1])

    @property
    def commanded_target(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._raw_actions.copy_(torch.nan_to_num(actions).clamp(-1.0, 1.0))
        limits = self._target_limits()
        self._processed_actions.copy_(
            bounded_target(
                self._processed_actions,
                self._raw_actions,
                limits,
                self._env.step_dt,
                self.rate_cfg.max_target_speed,
            )
        )

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        self._processed_actions[ids] = self._entity.data.joint_pos[ids][
            :, self._target_ids
        ]
        if getattr(self, "_named_target_bounds", None) is not None:
            self._processed_actions[ids] = self._clamp_target(
                self._processed_actions[ids], ids
            )
