"""
spot_dynamic_model_v2.py
========================
Dynamic model cua Boston Dynamics Spot - FIX rotational inertia.

THAY DOI SO VOI v1:
    - _mass_matrix() bo sung Angular Jacobian Jw cho moi link
    - M(q) = sum( m*Jv'*Jv  +  Jw'*R*I*R'*Jw )
    - M diagonal tang tu ~0.05 len ~0.15-0.30 (dung hon thuc te)

Parameters lay tu SpotLeg.proto & Spot.proto (giu nguyen).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ══════════════════════════════════════════════════════════════════
# 1.  INERTIAL PARAMETERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class LinkParams:
    mass: float
    com:  np.ndarray
    I:    np.ndarray

def _inertia_from_proto(Ixx, Iyy, Izz, Ixy, Ixz, Iyz) -> np.ndarray:
    return np.array([
        [Ixx,  Ixy,  Ixz],
        [Ixy,  Iyy,  Iyz],
        [Ixz,  Iyz,  Izz]
    ])

BODY_MASS = 14.40973585
BODY_COM  = np.array([0.022374, 0.0, 0.02032])

SHOULDER = LinkParams(
    mass = 2.131962264,
    com  = np.array([0.000643, -0.000985966468486107, 0.000873]),
    I    = _inertia_from_proto(
        Ixx =  0.0042655393382746935,
        Iyy =  0.004840458991120049,
        Izz =  0.004164087201524784,
        Ixy =  1.4537990385136328e-06,
        Ixz = -1.2872309568206006e-06,
        Iyz =  1.9737366790868247e-06
    )
)

UPPER_ARM = LinkParams(
    mass = 0.934886792,
    com  = np.array([0.11714932354424774, -0.12276425577719878, 0.08285389390894864]),
    I    = _inertia_from_proto(
        Ixx =  0.1012996403729267,
        Iyy =  0.09841956052568711,
        Izz =  0.06647528278171203,
        Ixy = -0.02395093549986319,
        Ixz = -0.04470786622876792,
        Iyz = -0.03546126467713879
    )
)

FOREARM = LinkParams(
    mass = 0.137150943,
    com  = np.array([0.0010379999999999999, -0.12989919535771882, -0.12095184442508211]),
    I    = _inertia_from_proto(
        Ixx =  0.014652279026458286,
        Iyy =  0.026034033109202114,
        Izz =  0.011480093961576602,
        Ixy =  6.107196767681762e-05,
        Ixz =  0.012746451515841613,
        Iyz = -6.558985205280841e-05
    )
)

LINKS = [SHOULDER, UPPER_ARM, FOREARM]

JOINT_LIMITS = {
    'abduction': (-0.6,   0.5),
    'rotation':  (-1.7,   1.7),
    'elbow':     (-0.45,  1.6),
}

HIP_POSITIONS = {
    'FL': np.array([ 0.3635,  0.0528, 0.0118]),
    'FR': np.array([ 0.3635, -0.0528, 0.0118]),
    'RL': np.array([-0.3084,  0.0528, 0.0117]),
    'RR': np.array([-0.3084, -0.0528, 0.0117]),
}

g   = 9.81
EPS = 1e-7

# ══════════════════════════════════════════════════════════════════
# 2.  ROTATION HELPERS
# ══════════════════════════════════════════════════════════════════

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def mirror_com(com: np.ndarray, side: str) -> np.ndarray:
    if side == 'L':
        com = com.copy(); com[0] = -com[0]
    return com

def mirror_inertia(I: np.ndarray, side: str) -> np.ndarray:
    if side == 'L':
        I = I.copy()
        I[0,1] = I[1,0] = -I[0,1]
        I[1,2] = I[2,1] = -I[1,2]
    return I

# ══════════════════════════════════════════════════════════════════
# 3.  FORWARD KINEMATICS
# ══════════════════════════════════════════════════════════════════

def fk_leg(q, hip_pos, side='R'):
    q1, q2, q3 = q
    sgn = -1 if side == 'L' else 1

    T0 = np.eye(4); T0[:3, 3] = hip_pos

    T1 = np.eye(4)
    T1[:3, :3] = Rz(q1)
    T1[:3,  3] = np.array([sgn * 0.0528, 0.0006, 0.0])

    T2 = np.eye(4)
    T2[:3, :3] = Rx(q2)
    T2[:3,  3] = np.array([0.0, -0.00053, 0.0])

    T3 = np.eye(4)
    T3[:3, :3] = Rx(q3)
    T3[:3,  3] = np.array([sgn * 0.1122, -0.319729, 0.182338])

    d_foot = np.array([sgn * 0.001038, 0.037459, 0.024742])

    T_sh = T0 @ T1
    T_up = T_sh @ T2
    T_lo = T_up @ T3

    p_foot = (T_lo @ np.append(d_foot, 1.0))[:3]
    return p_foot, [T_sh, T_up, T_lo]

# ══════════════════════════════════════════════════════════════════
# 4.  JACOBIAN (foot position)
# ══════════════════════════════════════════════════════════════════

def jacobian_leg(q, hip_pos, side='R'):
    p0, _ = fk_leg(q, hip_pos, side)
    J = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3); dq[j] = EPS
        p1, _ = fk_leg(q + dq, hip_pos, side)
        J[:, j] = (p1 - p0) / EPS
    return J

# ══════════════════════════════════════════════════════════════════
# 5.  DYNAMIC MODEL  M(q), C(q,q̇), G(q)
#     FIX: Bo sung rotational inertia vao M(q)
# ══════════════════════════════════════════════════════════════════

def _com_world(i, q, hip_pos, side):
    _, T_list = fk_leg(q, hip_pos, side)
    com = mirror_com(LINKS[i].com, side)
    return (T_list[i] @ np.append(com, 1.0))[:3]

def _get_rotation(i, q, hip_pos, side):
    """Lay ma tran rotation cua link i trong world frame."""
    _, T_list = fk_leg(q, hip_pos, side)
    return T_list[i][:3, :3]

def _jv(i, q, hip_pos, side):
    """Translational Jacobian cua CoM link i, shape 3x3."""
    p0 = _com_world(i, q, hip_pos, side)
    Jv = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3); dq[j] = EPS
        Jv[:, j] = (_com_world(i, q + dq, hip_pos, side) - p0) / EPS
    return Jv

def _jw(i, q, hip_pos, side):
    """
    Angular Jacobian cua link i, shape 3x3.
    Jw[:,j] = truc quay cua joint j trong world frame (neu j <= i).
    Joint axes:
        joint 0 (abduction): truc Z sau khi apply T0
        joint 1 (rotation):  truc X sau khi apply T0@T1
        joint 2 (elbow):     truc X sau khi apply T0@T1@T2
    """
    q1, q2, q3 = q
    sgn = -1 if side == 'L' else 1

    T0 = np.eye(4); T0[:3, 3] = HIP_POSITIONS.get(
        'FL' if side == 'L' else 'FR', hip_pos
    )
    # Su dung hip_pos truc tiep
    T0 = np.eye(4); T0[:3, 3] = hip_pos

    T1 = np.eye(4)
    T1[:3, :3] = Rz(q1)
    T1[:3,  3] = np.array([sgn * 0.0528, 0.0006, 0.0])

    T2 = np.eye(4)
    T2[:3, :3] = Rx(q2)
    T2[:3,  3] = np.array([0.0, -0.00053, 0.0])

    # Rotation matrix tai moi joint frame
    R0  = T0[:3, :3]                    # = I
    R01 = (T0 @ T1)[:3, :3]            # sau joint 0
    R012 = (T0 @ T1 @ T2)[:3, :3]      # sau joint 1

    # Truc quay trong world frame
    z_world = R0  @ np.array([0, 0, 1])   # joint 0: truc Z
    x1_world = R01 @ np.array([1, 0, 0])  # joint 1: truc X
    x2_world = R012 @ np.array([1, 0, 0]) # joint 2: truc X

    joint_axes = [z_world, x1_world, x2_world]

    Jw = np.zeros((3, 3))
    for j in range(3):
        if j <= i:   # chi joint truoc hoac bang link i moi anh huong
            Jw[:, j] = joint_axes[j]
    return Jw

def _mass_matrix_full(q, hip_pos, side):
    """
    M(q) day du: translational + rotational inertia.
    M = sum_i [ m_i * Jv_i' * Jv_i  +  Jw_i' * R_i * I_i * R_i' * Jw_i ]
    """
    M = np.zeros((3, 3))
    for i in range(3):
        Jv = _jv(i, q, hip_pos, side)
        Jw = _jw(i, q, hip_pos, side)
        R  = _get_rotation(i, q, hip_pos, side)
        I  = mirror_inertia(LINKS[i].I, side)

        # Translational term
        M += LINKS[i].mass * Jv.T @ Jv

        # Rotational term (FIX chinh)
        I_world = R @ I @ R.T
        M += Jw.T @ I_world @ Jw

    return M

def compute_MCG(q, dq, hip_pos, side='R'):
    """
    Tinh M(q), C(q,dq), G(q) cho mot chan 3-DOF.
    Version v2: M day du voi rotational inertia.
    """
    n     = 3
    g_vec = np.array([0.0, 0.0, -g])

    # M(q) - day du
    M = _mass_matrix_full(q, hip_pos, side)

    # G(q)
    G = np.zeros(n)
    for i in range(n):
        Jv = _jv(i, q, hip_pos, side)
        G -= LINKS[i].mass * g_vec @ Jv

    # C(q,dq) via Christoffel symbols
    M0  = M.copy()
    dM  = np.zeros((n, n, n))
    for k in range(n):
        dqk = np.zeros(n); dqk[k] = EPS
        dM[:, :, k] = (_mass_matrix_full(q + dqk, hip_pos, side) - M0) / EPS

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c_ijk = 0.5 * (dM[i,j,k] + dM[i,k,j] - dM[j,k,i])
                C[i, j] += c_ijk * dq[k]

    return M, C, G

# ══════════════════════════════════════════════════════════════════
# 6.  TOAN BO ROBOT 12 DOF
# ══════════════════════════════════════════════════════════════════

LEG_CONFIG = [
    ('FL', HIP_POSITIONS['FL'], 'L'),
    ('FR', HIP_POSITIONS['FR'], 'R'),
    ('RL', HIP_POSITIONS['RL'], 'L'),
    ('RR', HIP_POSITIONS['RR'], 'R'),
]

def compute_MCG_full(q_all, dq_all):
    M12 = np.zeros((12, 12))
    C12 = np.zeros((12, 12))
    G12 = np.zeros(12)
    for idx_leg, (_, hip_pos, side) in enumerate(LEG_CONFIG):
        s = slice(idx_leg*3, idx_leg*3 + 3)
        M_i, C_i, G_i = compute_MCG(q_all[s], dq_all[s], hip_pos, side)
        M12[s, s] = M_i
        C12[s, s] = C_i
        G12[s]    = G_i
    return M12, C12, G12

def clip_joints(q_all):
    q = q_all.copy()
    limits = [JOINT_LIMITS['abduction'], JOINT_LIMITS['rotation'], JOINT_LIMITS['elbow']]
    for leg in range(4):
        for j, (lo, hi) in enumerate(limits):
            q[leg*3 + j] = np.clip(q[leg*3 + j], lo, hi)
    return q

# ══════════════════════════════════════════════════════════════════
# 7.  SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True)
    print("=" * 60)
    print("Spot Dynamic Model v2 — With Rotational Inertia")
    print("=" * 60)

    q_stand = np.array([0., 0.7, 0.])
    dq_zero = np.zeros(3)

    M, C, G = compute_MCG(q_stand, dq_zero, HIP_POSITIONS['FL'], 'L')

    print(f"\nM diagonal (v2): {np.diag(M)}")
    print(f"G [Nm]:          {G}")

    eigs = np.linalg.eigvalsh(M)
    print(f"\nEigenvalues M:   {eigs}")
    print(f"Positive def?    {np.all(eigs > 0)}")
    print(f"Condition number: {eigs[-1]/eigs[0]:.1f}")

    print(f"\nSo sanh M diagonal:")
    print(f"  v1 (chi Jv):  [0.0552, 0.0488, 0.0043]")
    print(f"  v2 (Jv+Jw):   {np.diag(M)}")
    print(f"  Tang bao nhieu: {np.diag(M) / np.array([0.0552, 0.0488, 0.0043])}")

    total = BODY_MASS + 4*(SHOULDER.mass + UPPER_ARM.mass + FOREARM.mass)
    print(f"\nTong khoi luong: {total:.4f} kg")
    print("\nDynamic model v2 san sang cho SMC controller")
