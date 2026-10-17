"""Actions with a randomized control latency, for deployment-oriented training."""

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position targets that reach the actuators ``lag`` control steps late.

    ``lag`` is drawn per environment, uniformly from 0..max_lag, at every reset, so
    the policy sees the latency of a real control loop (sensing, computation, bus)
    without being able to read it.  The policy's own raw action is still what the
    ``last_action`` observation and the action-rate penalty use.
    """

    max_lag: int = 2

    def build(self, env):
        return DelayedJointPositionAction(self, env)


class DelayedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        if cfg.max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        # history[0] is the newest command; history[k] is k steps old.
        self._history = torch.zeros(
            cfg.max_lag + 1, self.num_envs, self.action_dim, device=self.device
        )
        self._lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._rows = torch.arange(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        count = self.num_envs if isinstance(ids, slice) else len(ids)
        self._history[:, ids] = 0.0
        self._lag[ids] = torch.randint(
            0, self.cfg.max_lag + 1, (count,), device=self._lag.device
        )

    def process_actions(self, actions):
        self._history = torch.roll(self._history, shifts=1, dims=0)
        self._history[0] = actions
        super().process_actions(self._history[self._lag, self._rows])
