import math
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

_ASSET_FORWARD_SIGN = 1.0


def _base(env):
    robot = env.scene["robot"]
    return robot, robot.find_bodies("base")[0][0]


def sampled_frequency(env, min_frequency=None, max_frequency=None):
    fixed = getattr(env.cfg, "test_frequency", None)
    if fixed is not None:
        current = torch.full((env.num_envs,), fixed, device=env.device)
        env._zbot_step_frequency = current
        return current
    min_frequency = (
        env.cfg.step_frequency_min if min_frequency is None else min_frequency
    )
    max_frequency = (
        env.cfg.step_frequency_max if max_frequency is None else max_frequency
    )
    current = getattr(env, "_zbot_step_frequency", None)
    if current is None or current.shape[0] != env.num_envs:
        current = torch.full((env.num_envs,), min_frequency, device=env.device)
    fresh = env.episode_length_buf == 0
    if fresh.any():
        if min_frequency == max_frequency:
            values = torch.full((int(fresh.sum()),), min_frequency, device=env.device)
        elif getattr(env.cfg, "frequency_sampling", "random") == "sequential":
            values = torch.linspace(
                min_frequency, max_frequency, env.num_envs, device=env.device
            )[fresh]
        else:
            values = torch.empty((int(fresh.sum()),), device=env.device).uniform_(
                min_frequency, max_frequency
            )
        current = current.clone()
        current[fresh] = values
    env._zbot_step_frequency = current
    return current


def phase(env, min_frequency=None, max_frequency=None):
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    return (
        env.episode_length_buf.float() * env.step_dt * frequency * 2.0 * math.pi
    ).remainder(2.0 * math.pi)


def phase_observation(env, min_frequency=None, max_frequency=None):
    p = phase(env, min_frequency, max_frequency)
    return torch.stack((torch.sin(p), torch.cos(p)), dim=-1)


def frequency_observation(env, min_frequency=None, max_frequency=None):
    """Expose the sampled episode frequency normalized to [0, 1]."""
    min_frequency = (
        env.cfg.step_frequency_min if min_frequency is None else min_frequency
    )
    max_frequency = (
        env.cfg.step_frequency_max if max_frequency is None else max_frequency
    )
    frequency = sampled_frequency(env, min_frequency, max_frequency)
    span = max(max_frequency - min_frequency, 1e-6)
    return ((frequency - min_frequency) / span).unsqueeze(-1)


def base_lin_vel(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, left_b, gravity_b = locomotion_frame(q)
    velocity_w = robot.data.body_link_lin_vel_w[:, i]
    forward_w = quat_apply(q, forward_b)
    left_w = quat_apply(q, left_b)
    gravity_w = quat_apply(q, gravity_b)
    return torch.stack(
        (
            (velocity_w * forward_w).sum(-1),
            (velocity_w * left_w).sum(-1),
            (velocity_w * gravity_w).sum(-1),
        ),
        dim=-1,
    )


def forward_velocity_reward(env, scale=0.2):
    """Bounded forward-speed reward with diminishing returns."""
    speed = base_lin_vel(env)[:, 0].clamp_min(0.0)
    return 1.0 - torch.exp(-speed / scale)


def lateral_velocity_penalty(env):
    """Penalize velocity projected onto the horizontal locomotion-left axis."""
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    _, left_b, _ = locomotion_frame(q)
    left_w = quat_apply(q, left_b)
    lateral_velocity = (robot.data.body_link_lin_vel_w[:, i] * left_w).sum(-1)
    return lateral_velocity.square()


class ForwardVelocityVisualizer(UniformVelocityCommand):
    """Viewer-only arrows for the Base forward axis and measured speed."""

    def _resample_command(self, env_ids):
        self.vel_command_b[env_ids] = 0.0

    def _debug_vis_impl(self, visualizer):
        indices = visualizer.get_env_indices(self.num_envs)
        if not indices:
            return
        robot, i = _base(self._env)
        q = robot.data.body_link_quat_w[:, i]
        forward_b, _, _ = locomotion_frame(q)
        forward_w = quat_apply(q, forward_b).cpu().numpy()
        pos = robot.data.body_link_pos_w[:, i].cpu().numpy()
        speed = base_lin_vel(self._env)[:, 0].detach().cpu().numpy()
        for j in indices:
            origin = pos[j] + np.array([0.0, 0.0, 0.12])
            scale = 0.25
            visualizer.add_arrow(
                origin,
                origin + forward_w[j] * scale,
                color=(0.0, 1.0, 1.0, 0.8),
                width=0.012,
            )
            visualizer.add_arrow(
                origin,
                origin + forward_w[j] * (speed[j] * scale),
                color=(0.1, 1.0, 0.2, 0.9),
                width=0.018,
            )


def command_arrow_segments(origin, forward, left, command, length=0.25):
    """Arrow segments for a normalized (vx, vy, wz) command in the Zbot frame.

    Returns ((start, end) of the planar-velocity arrow, (start, end) of the yaw
    arrow).  The yaw arrow is vertical: by the right-hand rule a positive
    (counter-clockwise seen from above) rate points up along world +Z.
    """
    vx, vy, wz = command
    up = np.array([0.0, 0.0, 1.0])
    linear = (origin, origin + (forward * vx + left * vy) * length)
    yaw = (origin, origin + up * wz * length)
    return linear, yaw


@dataclass(kw_only=True)
class VelocityCommandVizCfg(UniformVelocityCommandCfg):
    # Command magnitude that draws a full-length arrow on each axis (vx, vy, wz).
    full_scale: tuple[float, float, float] = (0.5, 0.2, 0.5)

    def build(self, env):
        return VelocityCommandVisualizer(self, env)


class VelocityCommandVisualizer(UniformVelocityCommand):
    """Viewer-only (vx, vy, wz) arrows above the base: blue command, green measured.

    Uses the Zbot forward/left axes rather than the floating foot_0 root frame.
    Lengths are normalized per axis by cfg.full_scale, so the three components
    are comparable even though their units differ.
    """

    def _debug_vis_impl(self, visualizer):
        indices = visualizer.get_env_indices(self.num_envs)
        if not indices:
            return
        robot, i = _base(self._env)
        q = robot.data.body_link_quat_w[:, i]
        forward_b, left_b, _ = locomotion_frame(q)
        forward = quat_apply(q, forward_b).cpu().numpy()
        left = quat_apply(q, left_b).cpu().numpy()
        pos = robot.data.body_link_pos_w[:, i].cpu().numpy()
        # base_ang_vel's third axis points down; negate for counter-clockwise > 0.
        measured = torch.cat(
            (base_lin_vel(self._env)[:, :2], -base_ang_vel(self._env)[:, 2:]), dim=-1
        )
        full = np.asarray(self.cfg.full_scale)
        commanded = self.command.cpu().numpy() / full
        measured = measured.cpu().numpy() / full
        for j in indices:
            origin = pos[j] + np.array([0.0, 0.0, 0.12])
            for values, color, width, lift in (
                (commanded[j], (0.2, 0.4, 1.0, 0.85), 0.012, 0.0),
                (measured[j], (0.1, 1.0, 0.2, 0.9), 0.018, 0.02),
            ):
                lifted = origin + np.array([0.0, 0.0, lift])
                for start, end in command_arrow_segments(
                    lifted, forward[j], left[j], values
                ):
                    if np.linalg.norm(end - start) > 1e-3:
                        visualizer.add_arrow(start, end, color=color, width=width)


def base_ang_vel(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, left_b, gravity_b = locomotion_frame(q)
    velocity_w = robot.data.body_link_ang_vel_w[:, i]
    forward_w = quat_apply(q, forward_b)
    left_w = quat_apply(q, left_b)
    gravity_w = quat_apply(q, gravity_b)
    return torch.stack(
        (
            (velocity_w * forward_w).sum(-1),
            (velocity_w * left_w).sum(-1),
            (velocity_w * gravity_w).sum(-1),
        ),
        dim=-1,
    )


def projected_gravity(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    g = torch.zeros((env.num_envs, 3), device=env.device)
    g[:, 2] = -1
    return quat_apply_inverse(q, g)


def upright(env, std=0.45):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device)
    z[:, 2] = 1
    z_w = quat_apply(q, z)
    return torch.exp(-z_w[:, 2].square() / std**2)


def locomotion_frame(quat):
    world_up = torch.zeros_like(quat[..., :3])
    world_up[..., 2] = 1.0
    gravity = -world_up
    body_z = torch.zeros_like(quat[..., :3])
    body_z[..., 2] = 1.0
    body_z_w = quat_apply(quat, body_z)
    body_z_w[..., 2] = 0.0
    body_z_w = body_z_w / body_z_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    forward_w = torch.linalg.cross(gravity, body_z_w)
    forward_w = forward_w * _ASSET_FORWARD_SIGN
    forward_w = forward_w / forward_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left_w = torch.linalg.cross(forward_w, gravity)
    left_w = left_w / left_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (
        quat_apply_inverse(quat, forward_w),
        quat_apply_inverse(quat, left_w),
        quat_apply_inverse(quat, gravity),
    )


def _forward_world(env):
    robot, i = _base(env)
    q = robot.data.body_link_quat_w[:, i]
    forward_b, _, _ = locomotion_frame(q)
    return quat_apply(q, forward_b)


def _yaw_error(env):
    current = _forward_world(env)
    initial = getattr(env, "_zbot_initial_forward_w", None)
    if initial is None or initial.shape != current.shape:
        initial = current.clone()
    fresh = env.episode_length_buf == 0
    initial = torch.where(fresh.unsqueeze(-1), current, initial)
    env._zbot_initial_forward_w = initial
    cross_z = current[:, 0] * initial[:, 1] - current[:, 1] * initial[:, 0]
    dot = (current[:, :2] * initial[:, :2]).sum(-1)
    return torch.atan2(cross_z, dot)


def yaw_error_observation(env):
    return _yaw_error(env).unsqueeze(-1)


def yaw_drift_penalty(env):
    """Penalize yaw drift from each episode's initial forward direction."""
    return _yaw_error(env).square()


def feet_contact(env, sensor_name="feet_ground_contact"):
    found = env.scene[sensor_name].data.found
    assert found is not None
    return (found > 0).float()


def alternating_foot_phase(
    env, sensor_name="feet_ground_contact", min_frequency=None, max_frequency=None
):
    p = phase(env, min_frequency, max_frequency)
    force = env.scene[sensor_name].data.force
    assert force is not None
    support = force.norm(dim=-1)
    left_force, right_force = support[:, 0], support[:, 1]
    total = left_force + right_force
    measured = (left_force - right_force) / total.clamp_min(1e-6)
    target = torch.sin(p)
    matched = 1.0 - (measured - target).square()
    return matched.clamp_min(0.0) * (total > 1e-3).float()


def swing_clearance(
    env,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
    target=0.05,
    min_frequency=None,
    max_frequency=None,
):
    del sensor_name
    p = phase(env, min_frequency, max_frequency)
    target_support = torch.sin(p)
    # alternating_foot_phase uses positive values for foot_0 support and
    # negative values for foot_1 support, so the opposite foot must swing.
    swing = torch.stack((target_support < 0, target_support > 0), dim=-1).float()
    heights = env.scene[height_sensor_name].data.heights
    return (swing * torch.exp(-((heights - target) / 0.04).square())).mean(-1)


def support_foot_slip(
    env,
    sensor_name="feet_ground_contact",
    site_names=("foot_0_sole", "foot_1_sole"),
    angular_radius=0.05,
):
    """Penalize translational and rotational slip while feet are in contact."""
    if angular_radius <= 0:
        raise ValueError("Angular radius must be positive")
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    site_ids, _ = robot.find_sites(site_names, preserve_order=True)
    if len(site_ids) != contact.shape[1]:
        raise ValueError("Expected one sole site per contact sensor primary")
    linear_speed_sq = robot.data.site_lin_vel_w[:, site_ids, :2].square().sum(-1)
    yaw_speed_sq = robot.data.site_ang_vel_w[:, site_ids, 2].square()
    speed_sq = linear_speed_sq + angular_radius**2 * yaw_speed_sq
    return (contact * speed_sq).sum(-1)


def soft_landing(env, sensor_name="feet_ground_contact"):
    """Penalize landing peak force and log the corresponding impulse."""
    sensor = env.scene[sensor_name]
    data = sensor.data
    assert data.force is not None
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    force_norm = data.force.norm(dim=-1)
    force_history = getattr(data, "force_history", None)
    if force_history is not None:
        history_norm = force_history.norm(dim=-1)
        peak_force = history_norm.amax(dim=-1)
        impulse = history_norm.sum(dim=-1) * env.physics_dt
    else:
        peak_force = force_norm
        impulse = force_norm * env.step_dt
    landing = first_contact.float()
    peak_landing_force = peak_force * landing
    landing_impulse = impulse * landing
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})["Metrics/landing_force_peak"] = (
            peak_landing_force.sum(-1).mean()
        )
        env.extras["log"]["Metrics/landing_impulse"] = landing_impulse.sum(-1).mean()
    return peak_landing_force.sum(-1)


def action_rate_l2(env):
    return (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)


def termination_penalty(env):
    """Return a one-time termination cost before reward dt scaling."""
    return envs_mdp.is_terminated(env) / env.step_dt


def joint_acc_l2(env):
    """Penalize the absolute sum of joint accelerations."""
    return env.scene["robot"].data.joint_acc.abs().sum(dim=-1)


def joint_pos_limits(env):
    r = env.scene["robot"].data
    return (r.joint_pos_limits[..., 0] - r.joint_pos).clamp_min(0).square().sum(-1) + (
        r.joint_pos - r.joint_pos_limits[..., 1]
    ).clamp_min(0).square().sum(-1)


def fallen(env, height=0.22, tilt=0.75):
    robot, i = _base(env)
    pos = robot.data.body_link_pos_w[:, i] - env.scene.env_origins
    q = robot.data.body_link_quat_w[:, i]
    z = torch.zeros((env.num_envs, 3), device=env.device)
    z[:, 2] = 1
    z_w = quat_apply(q, z)
    return (pos[:, 2] < height) | (z_w[:, 2].abs() > tilt)


class PartialBaseImpulse(envs_mdp.apply_body_impulse):
    """Run randomized impulses on a subset while leaving others disturbance-free."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        fraction = float(cfg.params.get("disturbed_fraction", 0.75))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("disturbed_fraction must be in [0, 1]")
        self._disturbed_mask = (
            torch.rand(self._num_envs, device=self._device) < fraction
        )

    def __call__(self, env, env_ids, *args, **kwargs):
        kwargs.pop("disturbed_fraction", None)
        super().__call__(env, env_ids, *args, **kwargs)
        quiet = (~self._disturbed_mask).nonzero(as_tuple=False).flatten()
        if quiet.numel():
            zeros = torch.zeros((len(quiet), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=quiet, body_ids=self._body_ids
            )

    def debug_vis(self, visualizer):
        """Draw only actual disturbed bodies, with arrows ending at the base."""
        viz = self._viz_cfg
        body_ids = self._body_ids
        if isinstance(body_ids, slice):
            body_ids = range(self._asset.num_bodies)[body_ids]
        for env_id in visualizer.get_env_indices(self._num_envs):
            if not self._active[env_id] or not self._disturbed_mask[env_id]:
                continue
            for body_id in body_ids:
                force = self._asset.data.body_external_wrench[env_id, body_id, :3]
                if force.norm().item() <= viz.min_force:
                    continue
                point = self._asset.data.body_com_pos_w[env_id, body_id]
                if self._body_point_offset is not None:
                    quat = self._asset.data.body_com_quat_w[env_id, body_id]
                    point = point + quat_apply(quat, self._body_point_offset)
                end = point.detach().cpu().numpy()
                start = end - force.detach().cpu().numpy() * viz.scale
                visualizer.add_arrow(start, end, color=viz.rgba, width=viz.width)


def foot_self_collision_penalty(env, sensor_name="feet_self_contact", scale=10.0):
    """Bounded penalty for contact force between the two feet."""
    if scale <= 0:
        raise ValueError("Self-collision scale must be positive")
    force = env.scene[sensor_name].data.force
    assert force is not None
    return 1.0 - torch.exp(-force.norm(dim=-1).sum(-1) / scale)


def foot_nonparallel_ground_penalty(env, sensor_name="feet_ground_contact", std=0.25):
    """Penalize contact feet whose sole normal is not aligned with world up."""
    if std <= 0:
        raise ValueError("Foot parallelism std must be positive")
    contact = feet_contact(env, sensor_name)
    robot = env.scene["robot"]
    sites = robot.find_sites(("foot_0_sole", "foot_1_sole"), preserve_order=True)[0]
    quat = robot.data.site_quat_w[:, sites]
    local_normals = torch.zeros_like(quat[..., :3])
    local_normals[..., 2] = 1.0
    normal_w = quat_apply(quat.reshape(-1, 4), local_normals.reshape(-1, 3)).reshape_as(
        local_normals
    )
    alignment = normal_w[..., 2].abs().clamp(0.0, 1.0)
    reward = torch.exp(-((1.0 - alignment) / std).square())
    return ((1.0 - reward) * contact).sum(-1) / contact.sum(-1).clamp_min(1.0)


def swing_foot_acceleration_penalty(
    env, acceleration_sensors, acceleration_scale, sensor_name="feet_ground_contact"
):
    """Bounded absolute world-frame acceleration cost for airborne feet only."""
    if acceleration_scale <= 0:
        raise ValueError("Acceleration scale must be positive")
    accelerations = torch.stack(
        [env.scene[name].data for name in acceleration_sensors], dim=1
    )
    cost = (accelerations.abs().sum(-1) / acceleration_scale).clamp(max=1.0)
    return (cost * (1.0 - feet_contact(env, sensor_name))).sum(-1)


def swing_foot_excess_height_penalty(
    env,
    max_height,
    height_scale,
    sensor_name="feet_ground_contact",
    height_sensor_name="foot_height_scan",
):
    """Penalize airborne foot clearance above the allowed terrain-relative height."""
    if max_height < 0 or height_scale <= 0:
        raise ValueError("Height threshold must be non-negative and scale positive")
    heights = env.scene[height_sensor_name].data.heights
    cost = ((heights - max_height).clamp_min(0.0) / height_scale).clamp(max=1.0)
    return (cost * (1.0 - feet_contact(env, sensor_name))).sum(-1)


def joint_acceleration_penalty(env, entity_name="robot"):
    """Sum absolute joint accelerations (rad/s²), without velocity differencing."""
    return env.scene[entity_name].data.joint_acc.abs().sum(dim=-1)


def joint_energy_penalty(env, energy_scale, entity_name="robot"):
    """Bounded mechanical-power cost: sum(abs(torque * joint velocity))."""
    if energy_scale <= 0:
        raise ValueError("Energy scale must be positive")
    asset = env.scene[entity_name]
    power = (asset.data.actuator_force * asset.data.joint_vel).abs().sum(-1)
    return (power / energy_scale).clamp(max=1.0)
