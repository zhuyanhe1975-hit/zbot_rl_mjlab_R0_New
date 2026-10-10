from pathlib import Path

import mujoco
import numpy as np
import pytest

from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS, JOINT_NAMES


@pytest.mark.parametrize("asset", ["robot.xml", "robot_geometry_inertia.xml"])
def test_sites_are_at_matching_sole_centers(asset):
    path = Path(__file__).resolve().parents[1] / "assets/zbot_6dof" / asset
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for name, position in zip(JOINT_NAMES, DEFAULT_JOINT_POS):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = position
    mujoco.mj_forward(model, data)
    sites = []
    for foot in ("foot_0", "foot_1"):
        geom_id = model.geom(foot + "_collisions").id
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start : start + count]
        world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
        world += data.geom_xpos[geom_id]
        sole = world[world[:, 2] < world[:, 2].min() + 0.0005]
        site = data.site_xpos[model.site(foot + "_sole").id]
        # Verify the point lies on the sole plane and near its horizontal center.
        assert abs(site[2] - sole[:, 2].mean()) < 0.0001
        center = (sole[:, :2].min(axis=0) + sole[:, :2].max(axis=0)) / 2
        np.testing.assert_allclose(site[:2], center, atol=0.001)
        sites.append(site.copy())
    assert abs(sites[0][2] - sites[1][2]) < 0.0001
