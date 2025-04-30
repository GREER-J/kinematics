import sympy as sp

# --- Time symbol ---
t = sp.symbols('t')

# --- State variables as functions of time ---
N, E, D = [sp.Function(x)(t) for x in ['N', 'E', 'D']]
phi, theta, psi = [sp.Function(x)(t) for x in ['phi', 'theta', 'psi']]
u, v, w = [sp.Function(x)(t) for x in ['u', 'v', 'w']]
p, q, r = [sp.Function(x)(t) for x in ['p', 'q', 'r']]

# --- Position and orientation ---
rBNn = sp.Matrix([[N], [E], [D]])
Theta = sp.Matrix([[phi], [theta], [psi]])
eta = sp.Matrix([[rBNn], [Theta]])

# --- Velocity vector ---
vBNb = sp.Matrix([[u], [v], [w]])
omegaBNb = sp.Matrix([[p], [q], [r]])
nu = sp.Matrix([[vBNb], [omegaBNb]])

# --- Rotation matrices ---
def Rx_sp(phi):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(phi), -sp.sin(phi)],
        [0, sp.sin(phi),  sp.cos(phi)]
    ])

def Ry_sp(theta):
    return sp.Matrix([
        [ sp.cos(theta), 0, sp.sin(theta)],
        [0, 1, 0],
        [-sp.sin(theta), 0, sp.cos(theta)]
    ])

def Rz_sp(psi):
    return sp.Matrix([
        [sp.cos(psi), -sp.sin(psi), 0],
        [sp.sin(psi),  sp.cos(psi), 0],
        [0, 0, 1]
    ])

# --- Rigid body rotation from body to inertial frame ---
Rnb = Rz_sp(psi) * Ry_sp(theta) * Rx_sp(phi)

# --- Angular velocity transformation matrix ---
Tk = sp.Matrix([
    [1, sp.sin(phi) * sp.tan(theta), sp.cos(phi) * sp.tan(theta)],
    [0, sp.cos(phi), -sp.sin(phi)],
    [0, sp.sin(phi) / sp.cos(theta), sp.cos(phi) / sp.cos(theta)]
])

# --- Velocity transformation matrix ---
Jk = sp.BlockMatrix([
    [Rnb, sp.zeros(3, 3)],
    [sp.zeros(3, 3), Tk]
]).as_explicit()

# --- Mass and inertia definitions ---
m, Ix, Iy, Iz = sp.symbols('m I_x I_y I_z')
I_body = sp.diag(Ix, Iy, Iz)

# --- Mass matrix ---
MRB = sp.BlockMatrix([
    [m * sp.eye(3), sp.zeros(3, 3)],
    [sp.zeros(3, 3), I_body]
]).as_explicit()

# --- Skew symmetric matrix ---
def skew(vec):
    return sp.Matrix([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])

# --- Coriolis matrix ---
S_v = skew(vBNb)
S_omega = skew(omegaBNb)

CRB = sp.BlockMatrix([
    [sp.zeros(3, 3), -m * S_v],
    [-m * S_v, -skew(I_body * omegaBNb)]
]).as_explicit()

# --- Gravity vector ---
g, z_g = sp.symbols('g z_g')
g_eta = sp.Matrix([
    [0],
    [0],
    [m * g],
    [m * g * z_g * sp.sin(theta)],
    [-m * g * z_g * sp.sin(phi) * sp.cos(theta)],
    [0]
])

# --- Generalized input force vector ---
tau = sp.Matrix(sp.symbols('tau1:7')).reshape(6, 1)

# --- Time derivative of velocity ---
nu_dot = sp.Matrix(sp.symbols('udot vdot wdot pdot qdot rdot')).reshape(6, 1)

# --- Full rigid-body dynamic equation ---
dyn_eq = MRB * nu_dot + CRB * nu + g_eta - tau

# Output
sp.pprint(dyn_eq, use_unicode=True)
