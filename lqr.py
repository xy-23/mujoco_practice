import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_are

import wip


x, dx, theta, dtheta = sp.symbols("x dx theta dtheta")
T = sp.symbols("T")

mB, mW, l, r, g = sp.symbols("mB mW l r g")
I2, J = sp.symbols("I2 J")

a = mB * l
IO = I2 + mB * l**2
m_tot = mB + 2 * mW
mO = m_tot + J / r**2
d1 = IO * mO - a**2 * sp.cos(theta) ** 2

ddx = (1 / d1) * (
    a * IO * dtheta**2 * sp.sin(theta)
    - a**2 * g * sp.sin(theta) * sp.cos(theta)
    + T * (IO / r + a * sp.cos(theta))
)

ddtheta = (1 / d1) * (
    -(a**2) * dtheta**2 * sp.sin(theta) * sp.cos(theta)
    + a * mO * g * sp.sin(theta)
    - T * (mO + a * sp.cos(theta) / r)
)

f = sp.Matrix([dx, ddx, dtheta, ddtheta])
A = f.jacobian([x, dx, theta, dtheta])
B = f.jacobian([T])

eq_point = {x: 0, dx: 0, theta: 0, dtheta: 0, T: 0}
A_lin = A.subs(eq_point)
B_lin = B.subs(eq_point)

wip_model = wip.make_model()
param_values = {
    mB: wip.TORSO_MASS,
    mW: wip.WHEEL_MASS * 2,
    l: wip.TORSO_SIZE[2],
    r: wip.WHEEL_SIZE[0],
    g: -wip_model.opt.gravity[2],
    I2: wip_model.body("torso").inertia[1],
    J: wip_model.body("left_wheel").inertia[2] * 2,
}
A_num = A_lin.subs(param_values).evalf()
B_num = B_lin.subs(param_values).evalf()
A = np.array(A_num, dtype=float)
B = np.array(B_num, dtype=float)

Q = np.diag([10, 10, 1, 1])
R = np.array([[100]])

S = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ (B.T @ S)
K = K.flatten()
