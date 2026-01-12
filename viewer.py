import time
import threading
import mujoco as mj
from collections.abc import Callable
from typing import override, TypeAlias

from PySide6.QtCore import QTimer, Qt
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QSurfaceFormat, QMouseEvent, QWheelEvent, QKeyEvent


# ref: https://gist.github.com/JeanElsner/755d0feb49864ecadab4ef00fd49a22b
format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2, 0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)

# set QSurfaceFormat before creating QApplication
app = QApplication()


class Viewer(QOpenGLWindow):

    Keys: TypeAlias = Qt.Key
    KeyEvent: TypeAlias = QKeyEvent

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *,
        fps: int = 60,
        step_per_ctrl: int = 1,
        ctrl: Callable[[None], None] = None,
        draw: Callable[[mj.MjrRect, mj.MjrContext], None] = None,
        key_press: Callable[[KeyEvent], None] = None,
        key_release: Callable[[KeyEvent], None] = None,
    ):
        super().__init__()

        self._w = None
        self._h = None
        self._s = None
        self._pos = None
        self.sim_running = False
        self.sim_thread = None

        self.model = model
        self.data = data
        self.opt = mj.MjvOption()
        self.scn = mj.MjvScene(model, maxgeom=10000)
        self.cam = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(model, self.cam)

        self.step_per_ctrl = step_per_ctrl
        self.ctrl = ctrl
        self.draw = draw
        self.key_press = key_press
        self.key_release = key_release

        self.timer = QTimer(interval=1000 // fps)
        self.timer.timeout.connect(self.update)  # this will call paintGL()
        self.timer.start()

    @override
    def mousePressEvent(self, event: QMouseEvent):
        self._pos = event.position()

    @override
    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mj.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self._pos.x()
        dy = pos.y() - self._pos.y()
        mj.mjv_moveCamera(
            self.model, action, dx / self._h, dy / self._h, self.scn, self.cam
        )
        self._pos = pos

    @override
    def wheelEvent(self, event: QWheelEvent):
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.0005 * event.angleDelta().y(),
            self.scn,
            self.cam,
        )

    @override
    def keyPressEvent(self, event: KeyEvent):
        if self.key_press is not None:
            self.key_press(event)

    @override
    def keyReleaseEvent(self, event: KeyEvent):
        if self.key_release is not None:
            self.key_release(event)

    @override
    def initializeGL(self):
        self.con = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_200)

    @override
    def resizeGL(self, w, h):
        self._w = w
        self._h = h
        self._s = self.devicePixelRatio()

    @override
    def paintGL(self) -> None:
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        viewport = mj.MjrRect(0, 0, int(self._w * self._s), int(self._h * self._s))
        mj.mjr_render(viewport, self.scn, self.con)
        if self.draw is not None:
            self.draw(viewport, self.con)

    def sim(self):
        t0 = time.time()

        while self.sim_running:
            sim_time = time.time() - t0

            while self.data.time < sim_time:
                mj.mj_step(self.model, self.data, self.step_per_ctrl)
                if self.ctrl is not None:
                    self.ctrl()

            time.sleep(0.001)

    def start_sim(self):
        if not self.sim_running:
            self.sim_running = True
            self.sim_thread = threading.Thread(target=self.sim, daemon=True)
            self.sim_thread.start()

    def stop_sim(self):
        if self.sim_running:
            self.sim_running = False
            if self.sim_thread is not None:
                self.sim_thread.join(timeout=1)

    def run(self):
        self.show()
        self.start_sim()
        app.exec()
        self.stop_sim()


if __name__ == "__main__":
    XML = """
    <mujoco>
        <visual>
            <rgba haze=".3 .3 .3 1"/>
        </visual>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.6 0.6 0.6" rgb2="0 0 0" width="512" height="512"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        </asset>
        <worldbody>
            <geom name="floor" pos="0 0 0" size="0 0 1" type="plane" material="matplane"/>
            <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
            <body pos="0 0 .5" euler="45 45 0">
                <freejoint/>
                <geom type="box" size=".1 .1 .1" rgba="1 1 1 .8" />
            </body>
        </worldbody>
    </mujoco>
    """

    model = mj.MjModel.from_xml_string(XML)
    data = mj.MjData(model)

    def draw(viewport: mj.MjrRect, con: mj.MjrContext):
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Time",
            f"{data.time:.3f}",
            con,
        )

    viewer = Viewer(model, data, draw=draw)
    viewer.resize(640, 480)
    viewer.run()
