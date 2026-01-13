import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

import wip
from lqr import K
from viewer import Viewer


class Controller:
    def __init__(self, model: mj.MjModel, data: mj.MjData, ctrl_dt: float):
        self.model = model
        self.data = data
        self.ctrl_dt = ctrl_dt

        self.x = 0
        self.v = 0
        self.pitch = 0
        self.vpitch = 0
        self.vyaw = 0

        self.xset = 0
        self.vset = 0
        self.vyawset = 0

        self.torque = 0
        self.turn = 0

        self.w_pressed = False
        self.a_pressed = False
        self.s_pressed = False
        self.d_pressed = False

    def ctrl(self):
        d = self.data

        q = d.body("torso").xquat
        gyro_data = d.sensor("gyro").data

        self.pitch = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[1]
        self.vpitch = gyro_data[1]

        # relative to torso
        left_motor_v = d.joint("left_wheel_joint").qvel[0]
        right_motor_v = d.joint("right_wheel_joint").qvel[0]

        # relative to world
        left_wheel_v = (left_motor_v + self.vpitch) * wip.WHEEL_SIZE[0]
        right_wheel_v = (right_motor_v + self.vpitch) * wip.WHEEL_SIZE[0]

        self.v = (left_wheel_v + right_wheel_v) / 2
        self.vset = (self.w_pressed - self.s_pressed) * 1

        self.vyaw = gyro_data[2]
        self.vyawset = (self.a_pressed - self.d_pressed) * 2

        self.x += self.v * self.ctrl_dt
        self.xset += self.vset * self.ctrl_dt

        self.torque = (
            K[0] * (self.xset - self.x)
            + K[1] * (self.vset - self.v)
            + K[2] * (0 - self.pitch)
            + K[3] * (0 - self.vpitch)
        ) / 2

        self.turn = (self.vyawset - self.vyaw) * 0.1

        d.ctrl[0] = self.torque - self.turn
        d.ctrl[1] = self.torque + self.turn

    def draw(self, viewport: mj.MjrRect, con: mj.MjrContext):

        theta = None
        for i in range(self.data.ncon):
            cont = self.data.contact[i]
            if (
                cont.geom1 == self.model.geom("floor_geom").id
                and cont.geom2 == self.model.geom("left_wheel_geom").id
            ):
                confrc = np.zeros(6, dtype=np.float64)
                mj.mj_contactForce(self.model, self.data, i, confrc)

                theta = np.atan2(confrc[2], confrc[0])
                break

        if theta is not None:
            w = self.data.joint("left_wheel_joint").qvel[0]
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                "t\ntheta\ntorque\nvset\nw",
                f"{self.data.time:.3f}\n{theta*57.3:.1f}\n{self.torque:.3f}\n{self.vset:.1f}\n{w:.3f}",
                con,
            )

    def key_press(self, event: Viewer.KeyEvent):
        if event.key() == Viewer.Keys.Key_W:
            self.w_pressed = True
        elif event.key() == Viewer.Keys.Key_S:
            self.s_pressed = True
        elif event.key() == Viewer.Keys.Key_A:
            self.a_pressed = True
        elif event.key() == Viewer.Keys.Key_D:
            self.d_pressed = True

    def key_release(self, event: Viewer.KeyEvent):
        if event.key() == Viewer.Keys.Key_W:
            self.w_pressed = False
        elif event.key() == Viewer.Keys.Key_S:
            self.s_pressed = False
        elif event.key() == Viewer.Keys.Key_A:
            self.a_pressed = False
        elif event.key() == Viewer.Keys.Key_D:
            self.d_pressed = False


if __name__ == "__main__":
    SIM_DT = 1 / 10000
    CTRL_DT = 1 / 100
    STEP_PER_CTRL = int(CTRL_DT / SIM_DT)

    model = wip.make_model()
    model.opt.timestep = SIM_DT
    model.vis.map.force = 0.1  # m/N

    data = mj.MjData(model)
    opt = mj.MjvOption()
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    controller = Controller(model, data, CTRL_DT)

    viewer = Viewer(
        model,
        data,
        opt=opt,
        step_per_ctrl=STEP_PER_CTRL,
        ctrl=controller.ctrl,
        draw=controller.draw,
        key_press=controller.key_press,
        key_release=controller.key_release,
    )
    viewer.resize(1280, 720)
    viewer.run()
