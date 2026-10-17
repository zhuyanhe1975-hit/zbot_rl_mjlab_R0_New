"""MJLab native viewer with a key that toggles a camera following the robot."""

import numpy as np
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import KEY_BACKSLASH, KEY_LEFT_BRACKET, KEY_RIGHT_BRACKET

from . import mdp

# MuJoCo and MJLab already bind most letters, so use a punctuation key.
FOLLOW_KEY = KEY_BACKSLASH
_FOLLOW_PAYLOAD = "zbot_follow_camera"


def follow_lookat(lookat, target):
    """Move the camera's look-at point over target in x/y, keeping its height."""
    moved = np.array(lookat, dtype=float)
    moved[:2] = np.asarray(target, dtype=float)[:2]
    return moved


class FollowCameraViewer(NativeMujocoViewer):
    """Press FOLLOW_KEY to make the camera follow the robot, press again to release.

    While following, the camera keeps the azimuth, elevation and distance you set
    with the mouse; only the look-at point tracks the robot (dragging to pan is
    overridden until you release).  Subclasses that replace sync_env_to_viewer must
    call _apply_follow() themselves.
    """

    def __init__(self, env, policy, **kwargs):
        super().__init__(env, policy, key_callback=self._on_key, **kwargs)
        self._follow = False

    def _on_key(self, key):
        # Runs on the MuJoCo viewer thread: only queue the request.
        if key == FOLLOW_KEY:
            self.request_action("CUSTOM", _FOLLOW_PAYLOAD)
        elif key in (KEY_RIGHT_BRACKET, KEY_LEFT_BRACKET):
            self.request_action("CUSTOM", "transform_stand" if key == KEY_RIGHT_BRACKET else "transform_snake")

    def _handle_custom_action(self, action, payload):
        if action == ViewerAction.CUSTOM and payload == _FOLLOW_PAYLOAD:
            return self._toggle_follow()
        if action == ViewerAction.CUSTOM and payload in ("transform_stand", "transform_snake"):
            env = self.env.unwrapped
            if "transform" not in env.cfg.commands:
                return False
            target = "stand" if payload == "transform_stand" else "snake"
            env.command_manager.get_term("transform").set_target(target)
            env.observation_manager.compute(update_history=True)
            print(f"[INFO] Transform command: {target}; manual control, automatic resets disabled")
            return True
        return super()._handle_custom_action(action, payload)

    def _toggle_follow(self):
        self._follow = not self._follow
        print(f"[INFO] Camera follow {'ON' if self._follow else 'OFF'}")
        return True

    def _follow_point(self):
        """World position the camera follows (the base of the selected env)."""
        robot, i = mdp._base(self.env.unwrapped)
        return robot.data.body_link_pos_w[self.env_idx, i].cpu().numpy()

    def _apply_follow(self):
        v = self.viewer
        if not self._follow or v is None or not v.is_running():
            return
        with v.lock():
            v.cam.lookat[:] = follow_lookat(v.cam.lookat, self._follow_point())

    def sync_env_to_viewer(self):
        self._apply_follow()
        super().sync_env_to_viewer()
