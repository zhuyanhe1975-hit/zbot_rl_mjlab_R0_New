"""Two-robot sim-to-sim viewer for Zbot MJLab tasks."""

import argparse
import pathlib
import sys
from copy import deepcopy

import mujoco
import mujoco.viewer
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.experiments.common import latest_checkpoint
from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS, robot_cfg
from zbot_rl_mjlab.viewer import FollowCameraViewer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    choices=[t for t in list_tasks() if t.startswith("Mjlab-Zbot")],
    default="Mjlab-Zbot-6dof-Velocity-Energy-Disturbed-Walking",
)
parser.add_argument("--checkpoint", type=pathlib.Path)
parser.add_argument("--frequency", type=float, default=0.5)
parser.add_argument("--target-speed", type=float, default=0.25)
parser.add_argument("--lateral-speed", type=float, default=0.0)
parser.add_argument("--yaw-rate", type=float, default=0.0)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--force-scale", type=float, default=1.0)
parser.add_argument("--torque-scale", type=float, default=1.0)
args = parser.parse_args()
TASK_ID = args.task
if args.force_scale < 0 or args.torque_scale < 0:
    raise ValueError("force-scale and torque-scale must be non-negative")
CHECKPOINT = args.checkpoint or latest_checkpoint(
    load_rl_cfg(TASK_ID).experiment_name, ROOT / "logs" / "rsl_rl"
)
print(f"Task: {TASK_ID}\nCheckpoint: {CHECKPOINT}\nFrequency: {args.frequency} Hz")

TASK_ENV_CFG = load_env_cfg(TASK_ID, play=True)
OBS_TERMS = tuple(TASK_ENV_CFG.observations["actor"].terms)
ACTION_SCALE = float(TASK_ENV_CFG.actions["joint_pos"].scale)
print(f"Action scale: {ACTION_SCALE}")

# Two copies of the robot in one scene: "ref" (MJLab, display only) and "native".
scene_cfg = deepcopy(TASK_ENV_CFG.scene)
scene_cfg.entities = {"ref": robot_cfg(), "native": robot_cfg()}
scene_cfg.sensors = ()
m = Scene(scene_cfg, device="cpu").compile()
TASK_ENV_CFG.sim.mujoco.apply(m)
d = mujoco.MjData(m)
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["actor_state_dict"]


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", ck["obs_normalizer._mean"])
        self.register_buffer("std", ck["obs_normalizer._std"])
        weights = {
            k: v
            for k, v in ck.items()
            if k.startswith("mlp.") and k.endswith(".weight")
        }
        layers = []
        for index, name in enumerate(
            sorted(weights, key=lambda x: int(x.split(".")[1]))
        ):
            out_dim, in_dim = weights[name].shape
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if index < len(weights) - 1:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers)
        self.net.load_state_dict(
            {k[4:]: v for k, v in ck.items() if k.startswith("mlp.")}
        )

    def forward(self, x):
        return self.net((x - self.mean) / (self.std + 1e-2))


actor = Actor().eval()
default = np.array(DEFAULT_JOINT_POS, dtype=np.float64)
STEP_DT = TASK_ENV_CFG.sim.mujoco.timestep * TASK_ENV_CFG.decimation
FREQ_MIN = TASK_ENV_CFG.step_frequency_min
FREQ_MAX = TASK_ENV_CFG.step_frequency_max


def adr(prefix):
    return [
        m.jnt_qposadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/joint{i}")
        ]
        for i in range(1, 7)
    ]


def dof(prefix):
    return [
        m.jnt_dofadr[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/joint{i}")
        ]
        for i in range(1, 7)
    ]


def body(prefix):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}/base")


rq, nq = adr("ref"), adr("native")
rd, nd = dof("ref"), dof("native")
rb, nb = body("ref"), body("native")


def locomotion_frame(r):
    """Mirror mdp.locomotion_frame: forward = gravity x body_z_h, left = forward x gravity."""
    gravity = np.array([0.0, 0.0, -1.0])
    z = r[:, 2].copy()
    z[2] = 0
    z /= max(np.linalg.norm(z), 1e-6)
    forward = np.cross(gravity, z)
    forward /= max(np.linalg.norm(forward), 1e-6)
    left = np.cross(forward, gravity)
    left /= max(np.linalg.norm(left), 1e-6)
    return forward, left, gravity


obs_sizes = []  # per-term dimensions, filled by obs()


def obs(b, qadr, dadr, last, step):
    """Rebuild the actor observation in the same term order as the MJLab task."""
    r = d.xmat[b].reshape(3, 3)
    vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, vel, 0)
    lin_w = vel[3:] + np.cross(vel[:3], d.xpos[b] - d.xipos[b])
    axes = locomotion_frame(r)
    forward = axes[0]
    initial = initial_forward
    phase = step * STEP_DT * args.frequency * 2 * np.pi
    values = {
        "base_lin_vel": [lin_w @ axis for axis in axes],
        "base_ang_vel": [vel[:3] @ axis for axis in axes],
        "projected_gravity": r.T @ np.array([0.0, 0.0, -1.0]),
        "yaw_error": [
            np.arctan2(
                forward[0] * initial[1] - forward[1] * initial[0],
                forward[:2] @ initial[:2],
            )
        ],
        "joint_pos": d.qpos[qadr] - default,
        "joint_vel": d.qvel[dadr],
        "actions": last,
        "phase": [np.sin(phase), np.cos(phase)],
        "frequency": [(args.frequency - FREQ_MIN) / max(FREQ_MAX - FREQ_MIN, 1e-6)],
        "target_speed": [args.target_speed],
        "velocity_command": [args.target_speed, args.lateral_speed, args.yaw_rate],
    }
    missing = [t for t in OBS_TERMS if t not in values]
    if missing:
        raise KeyError(f"Native observation has no builder for terms: {missing}")
    parts = [np.ravel(values[t]) for t in OBS_TERMS]
    obs_sizes[:] = [p.size for p in parts]
    value = np.concatenate(parts).astype(np.float32)
    if value.size != int(ck["obs_normalizer._mean"].numel()):
        raise ValueError(
            "Native observation dimension "
            f"{value.size} does not match training config/checkpoint "
            f"{ck['obs_normalizer._mean'].numel()}"
        )
    return value


# external MJLab owner
a = TASK_ENV_CFG
a.test_frequency = args.frequency
a.scene.num_envs = 1
if "target_speed" in a.commands:
    ranges = a.commands["target_speed"].ranges
    ranges.lin_vel_x = (args.target_speed,) * 2
    ranges.lin_vel_y = (args.lateral_speed,) * 2
    ranges.ang_vel_z = (args.yaw_rate,) * 2
    print(
        f"Command: vx {args.target_speed} m/s, vy {args.lateral_speed} m/s, "
        f"wz {args.yaw_rate} rad/s"
    )
if "base_disturbance" in a.events:
    params = a.events["base_disturbance"].params
    params["disturbed_fraction"] = 1.0
    params["force_range"] = tuple(v * args.force_scale for v in params["force_range"])
    params["torque_range"] = tuple(
        v * args.torque_scale for v in params["torque_range"]
    )
    print(
        f"Disturbance force scale: {args.force_scale}; torque scale: {args.torque_scale}"
    )
env = ManagerBasedRlEnv(a, device=args.device)
agent = load_rl_cfg(TASK_ID)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device=args.device)
runner.load(
    str(CHECKPOINT), load_cfg={"actor": True}, strict=True, map_location=args.device
)
policy = runner.get_inference_policy(device=args.device)


def root_adr(prefix):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}/root")
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]


NATIVE_ROOT_Q, _ = root_adr("native")
REF_ROOT_Q, REF_ROOT_D = root_adr("ref")
DISTURBED = "base_disturbance" in a.events
ENV_BASE = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/base")
initial_forward = np.zeros(3)
last = np.zeros(6)
native_step = 0


def place_ref():
    """Show the MJLab robot (display only) beside the native one."""
    src = env.sim.data.qpos[0].detach().cpu().numpy()
    d.qpos[REF_ROOT_Q : REF_ROOT_Q + 7] = src[:7] + np.array([0, 0.5, 0, 0, 0, 0, 0])
    d.qpos[rq] = src[7:13]
    # MuJoCo must never advance the reference body on its own.
    d.qvel[REF_ROOT_D : REF_ROOT_D + 6] = 0
    d.qvel[rd] = 0


def reset_native():
    """Start the native robot from the MJLab reset pose, 1 m to the side."""
    global initial_forward, last, native_step
    mujoco.mj_resetData(m, d)
    reset_qpos = env.sim.data.qpos[0].detach().cpu().numpy()
    d.qpos[NATIVE_ROOT_Q : NATIVE_ROOT_Q + 7] = reset_qpos[:7] + np.array(
        [0, -0.5, 0, 0, 0, 0, 0]
    )
    d.qpos[nq] = default
    place_ref()
    mujoco.mj_forward(m, d)
    initial_forward = locomotion_frame(d.xmat[nb].reshape(3, 3))[0]
    last = np.zeros(6)
    native_step = 0


def check_obs(native, reference, atol=1e-3):
    """Both robots start in the same state, so the first observations must match."""
    start = 0
    for name, size in zip(OBS_TERMS, obs_sizes):
        err = np.abs(native[start : start + size] - reference[start : start + size])
        if err.max() > atol:
            print(f"[WARN] native obs '{name}' differs from MJLab by {err.max():.4f}")
        start += size


def paired_policy(mj_obs):
    """MJLab action for the env, plus one native control step on the side."""
    global last, native_step
    ref_action = policy(mj_obs)
    native_obs = obs(nb, nq, nd, last, native_step)
    if native_step == 0:
        check_obs(native_obs, mj_obs["actor"][0].cpu().numpy())
    native_action = actor(torch.from_numpy(native_obs)[None]).numpy()[0]
    last = native_action.copy()
    native_step += 1
    place_ref()
    d.ctrl[:] = 0
    for i in range(6):
        d.ctrl[6 + i] = default[i] + ACTION_SCALE * native_action[i]
    for _ in range(a.decimation):
        mujoco.mj_step(m, d)
    return ref_action


class CompareViewer(FollowCameraViewer):
    """mjlab's native viewer (RT overlay, pause, speed keys) on the two-robot scene."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, enable_perturbations=False, **kwargs)

    def setup(self):
        self.mjm, self.mjd = m, d
        self.vopt = mujoco.MjvOption()
        self.viewer = mujoco.viewer.launch_passive(
            m,
            d,
            key_callback=self._safe_key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
        v = self.viewer
        # User-scene label geoms stay in the world and identify each controller.
        v.user_scn.ngeom = 2
        for geom, label, color in zip(
            v.user_scn.geoms[:2],
            ("pytorch+mujoco", "mjlab"),
            ((0.2, 0.7, 1.0, 1.0), (0.2, 1.0, 0.4, 1.0)),
        ):
            geom.type = mujoco.mjtGeom.mjGEOM_LABEL
            geom.size[:] = (0.12, 0.12, 0.12)
            geom.rgba[:] = color
            geom.label = label
        v.cam.distance = 2.2
        v.cam.azimuth = 135
        v.cam.elevation = -20
        v.cam.lookat[:] = [0, -0.06, 0.2]
        reset_native()

    def reset_environment(self):
        super().reset_environment()
        reset_native()

    def _follow_point(self):
        # Keep both robots in view: follow the midpoint between them.
        return (d.xpos[nb] + d.xpos[rb]) / 2

    def sync_viewer_to_env(self):
        pass  # No mouse perturbation on the combined scene.

    def sync_env_to_viewer(self):
        v = self.viewer
        if not v.is_running():
            return
        with self._mj_lock:
            place_ref()
            mujoco.mj_forward(m, d)
            self._apply_follow()
            if not v.is_running():
                return
            self._set_status_overlay(v)
            v.user_scn.ngeom = 2
            if DISTURBED:
                # Draw the active MJLab disturbance on the reference robot in pink.
                force = (
                    env.sim.data.xfrc_applied[0, ENV_BASE, :3].detach().cpu().numpy()
                )
                if np.linalg.norm(force) > 1e-5:
                    start = d.xpos[rb].copy()
                    geom = v.user_scn.geoms[v.user_scn.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_ARROW.value,
                        np.zeros(3),
                        np.zeros(3),
                        np.zeros(9),
                        np.array((0.95, 0.05, 0.75, 1.0), dtype=np.float32),
                    )
                    mujoco.mjv_connector(
                        geom,
                        mujoco.mjtGeom.mjGEOM_ARROW.value,
                        0.025,
                        start,
                        start + force * 0.04,
                    )
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    v.user_scn.ngeom += 1
            for geom, b in zip(v.user_scn.geoms[:2], (nb, rb)):
                geom.pos[:] = [d.xpos[b, 0], d.xpos[b, 1], d.xpos[b, 2] + 0.24]
            v.sync()


wrapped.reset()
try:
    CompareViewer(wrapped, paired_policy).run()
finally:
    env.close()
