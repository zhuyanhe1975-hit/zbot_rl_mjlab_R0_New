# ZBot 6DOF asset

`robot.xml` is a standalone conversion of the original project's
`assets/zbot_usd/zbot/zbot_6s_new.usd`. All training/runtime asset files are here;
the original project and Isaac Lab are not runtime dependencies.

Rebuild from the sibling project (conversion only needs Pixar USD Python):

```bash
/home/yhzhu/isaaclab/_isaac_sim/python.sh scripts/convert_zbot_usd.py
```

The converter also accepts `--source PATH --output DIR`. `provenance.json`
records the source SHA-256, physical properties, source zero poses and initial
walking state. Twelve rigid links total **3.00504004955 kg**, with six unlimited
hinges named **joint1 through joint6**. The free root is **foot_0**, matching the
USD ArticulationRootAPI; the observation body is **base**, and the opposite foot
is **foot_1**. A foot's body origin is not necessarily its bottom contact plane.

The original tree, mesh-to-body transforms, explicit mass, center of mass,
principal inertias, and joint axes/signs are preserved. Fixed links remain named
bodies. Adjacent-link collision exclusions follow the USD joint settings.
MuJoCo uses convex hull contacts for these collision meshes; this is not an
exact reproduction of PhysX convex decomposition or its contact solver.
Rendering uses simple blue/gray colors instead of Isaac MDL materials.

No ground or actuators are embedded: the task supplies ground, PD gains and
friction. XML zero joint position is the authored straight chain. Task reset
sets root position `(0, -0.06, 0)`, identity rotation, and ordered joint angles
`[0.312, 0.837, -2.02, 2.02, -0.837, -0.312]` radians. This puts the `base`
origin about 0.2545 m above the ground.

## Canonical inertia choice

`robot_geometry_inertia.xml` is the canonical asset used by training, playback and
visualization. It preserves each USD body's mass but recomputes the inertia from
the visual mesh geometry, so the inertia frame follows the assembled module
geometry. `robot.xml` remains available for comparison with the authored USD
principal axes. `inertia_comparison.json` records both variants.
