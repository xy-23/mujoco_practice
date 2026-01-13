import mujoco as mj


TORSO_SIZE = [0.061 / 2, 0.11 / 2, 0.15 / 2]
WHEEL_SIZE = [0.034, 0.026 / 2, 0]
WHEEL_OFFSET_Y = TORSO_SIZE[1] + WHEEL_SIZE[1]
TORSO_MASS = 1.2
WHEEL_MASS = 0.04

spec = mj.MjSpec()

spec.add_texture(
    name="floor_texture",
    type=mj.mjtTexture.mjTEXTURE_2D,
    builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
    mark=mj.mjtMark.mjMARK_EDGE,
    rgb1=[0.2, 0.3, 0.4],
    rgb2=[0.1, 0.2, 0.3],
    markrgb=[0.8, 0.8, 0.8],
    height=300,
    width=300,
)
floor_material = spec.add_material(
    name="floor_material",
    texuniform=True,
    texrepeat=[5, 5],
    reflectance=0.2,
)
floor_material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "floor_texture"

spec.add_texture(
    name="wheel_texture",
    type=mj.mjtTexture.mjTEXTURE_2D,
    builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
    mark=mj.mjtMark.mjMARK_EDGE,
    rgb1=[0.0, 0.0, 0.0],
    rgb2=[0.3, 0.3, 0.3],
    height=300,
    width=300,
)
wheel_material = spec.add_material(
    name="wheel_material",
    texuniform=True,
    texrepeat=[5, 5],
)
wheel_material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "wheel_texture"

spec.worldbody.add_geom(
    name="floor_geom",
    type=mj.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 1],
    material="floor_material",
)

torso = spec.worldbody.add_body(
    name="torso",
    pos=[0, 0, TORSO_SIZE[2] + WHEEL_SIZE[0]],
    euler=[0, 0, 0],
)
torso.add_geom(
    type=mj.mjtGeom.mjGEOM_BOX,
    size=TORSO_SIZE,
    mass=TORSO_MASS,
)
torso.add_site(name="imu")
torso.add_freejoint()
torso.add_camera(
    name="back",
    pos=[-1, 0, 0.5],
    xyaxes=[0, -1, 0, 1, 0, 3],
    mode=mj.mjtCamLight.mjCAMLIGHT_TRACKCOM,
)
torso.add_camera(
    name="side",
    pos=[0, -1, 0.5],
    xyaxes=[1, 0, 0, 0, 1, 3],
    mode=mj.mjtCamLight.mjCAMLIGHT_TRACKCOM,
)
torso.add_light(
    pos=[0, 0, 10],
    mode=mj.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    castshadow=False,
)

left_wheel = torso.add_body(
    name="left_wheel",
    pos=[0, WHEEL_OFFSET_Y, -TORSO_SIZE[2]],
    euler=[-90, 0, 0],
)
left_wheel.add_geom(
    name="left_wheel_geom",
    type=mj.mjtGeom.mjGEOM_CYLINDER,
    size=WHEEL_SIZE,
    mass=WHEEL_MASS,
    material="wheel_material",
)
left_wheel.add_joint(
    name="left_wheel_joint",
    type=mj.mjtJoint.mjJNT_HINGE,
    axis=[0, 0, 1],
)

right_wheel = torso.add_body(
    name="right_wheel",
    pos=[0, -WHEEL_OFFSET_Y, -TORSO_SIZE[2]],
    euler=[-90, 0, 0],
)
right_wheel.add_geom(
    name="right_wheel_geom",
    type=mj.mjtGeom.mjGEOM_CYLINDER,
    size=WHEEL_SIZE,
    mass=WHEEL_MASS,
    material="wheel_material",
)
right_wheel.add_joint(
    name="right_wheel_joint",
    type=mj.mjtJoint.mjJNT_HINGE,
    axis=[0, 0, 1],
)

spec.add_actuator(
    name="left_wheel_motor",
    trntype=mj.mjtTrn.mjTRN_JOINT,
    target="left_wheel_joint",
)
spec.add_actuator(
    name="right_wheel_motor",
    trntype=mj.mjtTrn.mjTRN_JOINT,
    target="right_wheel_joint",
)
spec.add_sensor(
    name="gyro",
    type=mj.mjtSensor.mjSENS_GYRO,
    objtype=mj.mjtObj.mjOBJ_SITE,
    objname="imu",
)


def make_model() -> mj.MjModel:
    return spec.compile()


if __name__ == "__main__":
    import mujoco.viewer as viewer

    viewer.launch(make_model())
