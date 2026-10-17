"""Three stochastic controls for six antisymmetric joint targets."""

from dataclasses import dataclass

import torch

from ...robot import JOINT_NAMES
from .actions import JointTargetRateAction, JointTargetRateActionCfg, bounded_target


def expand_mirror(values):
    return torch.cat((values, -values.flip(-1)), dim=-1)


def project_mirror(values):
    return (values[..., :3] - values[..., 3:].flip(-1)) * 0.5


def mirrored_last_action(env):
    # Preserve the pretrained 30-D observation schema, including six last-action slots.
    return expand_mirror(env.action_manager.get_term("joint_pos").raw_action)


@dataclass(kw_only=True)
class MirroredJointTargetRateActionCfg(JointTargetRateActionCfg):
    def build(self, env):
        return MirroredJointTargetRateAction(self, env)


class MirroredJointTargetRateAction(JointTargetRateAction):
    def __init__(self, cfg, env):
        self._mirror_target_limits = None
        # Let the parent allocate six actuator-target slots before reducing raw actions.
        super().__init__(cfg, env)
        if tuple(self._target_names) != JOINT_NAMES:
            raise ValueError("Mirror actions require joint1..joint6 in model order")
        self._action_dim = 3
        self._raw_actions = torch.zeros((env.num_envs, 3), device=env.device)
        self._paired_limits(validate=True)
        self._processed_actions.copy_(self._clamp_target(self._processed_actions))
        self._previous_commanded_target = self.commanded_target.clone()

    @property
    def commanded_target_velocity(self):
        return (self.commanded_target - self._previous_commanded_target) / self._env.step_dt

    def _paired_limits(self, validate=False):
        # Limits are configured at construction, not randomized during rollout.
        if self._mirror_target_limits is None:
            limits = self._target_limits()
            left, right = limits[:, :3], limits[:, 3:].flip(1)
            self._mirror_target_limits = torch.stack((torch.maximum(left[..., 0], -right[..., 1]), torch.minimum(left[..., 1], -right[..., 0])), dim=-1)
        paired = self._mirror_target_limits
        if validate and (paired[..., 0] > paired[..., 1]).any():
            raise ValueError("Joint limits have no common mirrored target interval")
        return paired

    def _clamp_target(self, target, env_ids=slice(None)):
        limits = self._paired_limits()[env_ids]
        coordinates = project_mirror(target).clamp(min=limits[..., 0], max=limits[..., 1])
        return expand_mirror(coordinates)

    def process_actions(self, actions):
        if actions.shape[-1] != 3:
            raise ValueError("Mirrored policy must provide exactly three actions")
        self._previous_commanded_target.copy_(self.commanded_target)
        self._raw_actions.copy_(torch.nan_to_num(actions).clamp(-1, 1))
        coordinates = bounded_target(project_mirror(self.commanded_target), self._raw_actions, self._paired_limits(), self._env.step_dt, self.rate_cfg.max_target_speed)
        self._processed_actions.copy_(expand_mirror(coordinates))

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._processed_actions[ids] = self._clamp_target(self._processed_actions[ids], ids)
        self._previous_commanded_target[ids] = self.commanded_target[ids]


def migrate_beta_mirror_head(actor_state):
    """Merge a native six-action Beta head into three mirrored Beta variables.

    The raw alpha-left / beta-right and beta-left / alpha-right rows are averaged.
    This is a linear-head projection, not exact preservation of the old nonlinear
    Beta means or its asymmetric behavior. Backbone and normalization are unchanged.
    """
    key = "mlp.4.weight"
    bias = "mlp.4.bias"
    if key not in actor_state or bias not in actor_state:
        raise ValueError("Expected the pretrained 256/128 MLP Beta action head")
    weight, offset = actor_state[key], actor_state[bias]
    if weight.shape[0] == 6 and offset.shape[0] == 6:
        return actor_state, False
    if weight.shape[0] != 12 or offset.shape[0] != 12:
        raise ValueError("Expected a six-action or three-action Beta checkpoint")
    left = torch.tensor([0, 1, 2, 6, 7, 8], device=weight.device)
    opposite = torch.tensor([11, 10, 9, 5, 4, 3], device=weight.device)
    migrated = dict(actor_state)
    migrated[key] = (weight[left] + weight[opposite]) * 0.5
    migrated[bias] = (offset[left] + offset[opposite]) * 0.5
    return migrated, True
