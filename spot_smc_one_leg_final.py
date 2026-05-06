"""
spot_smc_one_leg_final.py
=========================
SMC controller dung DUNG cong thuc theo yeu cau:

    tau = M(q)*(ddq_d + lambda*e_dot) + C(q,dq)*dq + G(q) + K*sat(s/phi)
    s   = e_dot + lambda*e

Dynamic model lay tu spot_dynamic_model.py (Task 3).
FL leg dung SMC torque control.
3 chan con lai giu position mode de body on dinh.

Chay:
    cd ~/ros2_ws/src/spot_smc_controller/spot_smc_controller/
    python3 spot_smc_one_leg_final.py
"""

from controller import Robot
import numpy as np
import os, json, sys

sys.path.insert(0, os.path.dirname(__file__))
from spot_dynamic_model import compute_MCG, HIP_POSITIONS

# ══════════════════════════════════════════════════════════════════
# SMC PARAMETERS
# ══════════════════════════════════════════════════════════════════
# M(q) diagonal ~ [0.001, 0.063, 0.076]  (tu test task 3)
# => tau_nominal = M * (ddq_d + lambda*e_dot) nho
# => K phai du lon de switching term bu uncertainty + gravity
#
# Tuning logic:
#   - LAMBDA: slope sliding surface. Start 2.0
#   - K: switching gain. Can ~10-20 de but gravity (~3-6 Nm)
#   - PHI: boundary layer. 0.1 de giam chattering

LAMBDA       = 2.0
K            = 15.0
PHI          = 0.1
TORQUE_LIMIT = 40.0
TIMESTEP     = 32      # ms

# Target: standing pose (elbow=0 trong joint limit [-0.45, 1.6])
Q_DESIRED   = np.array([0.0,  0.7,  0.0])
DQ_DESIRED  = np.zeros(3)
DDQ_DESIRED = np.zeros(3)

# FL leg config
LEG_SIDE = 'L'
HIP_POS  = HIP_POSITIONS['FL']

# Webots device names
FL_MOTOR_NAMES = [
    'front left shoulder abduction motor',
    'front left shoulder rotation motor',
    'front left elbow motor',
]
FL_SENSOR_NAMES = [
    'front left shoulder abduction sensor',
    'front left shoulder rotation sensor',
    'front left elbow sensor',
]

# 3 chan con lai giu standing pose (position mode)
OTHER_LEGS = {
    'FR': (['front right shoulder abduction motor',
             'front right shoulder rotation motor',
             'front right elbow motor'],
            [0.0, 0.7, 0.0]),
    'RL': (['rear left shoulder abduction motor',
             'rear left shoulder rotation motor',
             'rear left elbow motor'],
            [0.0, 0.7, 0.0]),
    'RR': (['rear right shoulder abduction motor',
             'rear right shoulder rotation motor',
             'rear right elbow motor'],
            [0.0, 0.7, 0.0]),
}

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def sat(x, phi):
    """Saturation function thay sign() - giam chattering."""
    return np.clip(x / phi, -1.0, 1.0)


class VelocityEstimator:
    """Numerical diff + low-pass filter de tinh dq tu q."""
    def __init__(self, n, alpha=0.3):
        self.q_prev  = None
        self.dq_filt = np.zeros(n)
        self.alpha   = alpha
        self.dt      = TIMESTEP / 1000.0

    def update(self, q):
        if self.q_prev is None:
            self.q_prev = q.copy()
            return np.zeros(len(q))
        dq_raw       = (q - self.q_prev) / self.dt
        self.dq_filt = self.alpha * dq_raw + (1 - self.alpha) * self.dq_filt
        self.q_prev  = q.copy()
        return self.dq_filt.copy()


# ══════════════════════════════════════════════════════════════════
# SMC TORQUE — DUNG CONG THUC CHINH XAC
# ══════════════════════════════════════════════════════════════════

def compute_smc_torque(q, dq):
    """
    Tinh torque theo dung cong thuc SMC:
        e      = q_d - q
        e_dot  = dq_d - dq
        s      = e_dot + lambda*e
        tau    = M(q)*(ddq_d + lambda*e_dot) + C(q,dq)*dq + G(q) + K*sat(s/phi)

    M, C, G lay tu spot_dynamic_model.py (Task 3).
    """
    # Lay dynamic model tu Task 3
    M, C, G = compute_MCG(q, dq, HIP_POS, LEG_SIDE)

    # Tracking error
    e     = Q_DESIRED  - q
    e_dot = DQ_DESIRED - dq

    # Sliding surface
    s = e_dot + LAMBDA * e

    # Nominal term: compensation chieu huong mong muon
    tau_nominal  = M @ (DDQ_DESIRED + LAMBDA * e_dot) + C @ dq + G

    # Switching term: robust compensation cho uncertainty
    tau_switching = K * sat(s, PHI)

    tau = tau_nominal + tau_switching
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    robot = Robot()
    dt    = TIMESTEP

    # ── 3 chan con lai: position mode de giu body on dinh
    for leg_name, (mnames, stand_q) in OTHER_LEGS.items():
        for i, mn in enumerate(mnames):
            m = robot.getDevice(mn)
            m.setPosition(stand_q[i])
            m.setVelocity(1.0)

    # ── FL leg: torque mode (SMC)
    fl_motors = []
    for mn in FL_MOTOR_NAMES:
        m = robot.getDevice(mn)
        m.setPosition(float('inf'))   # disable position control
        m.setVelocity(10.0)           # phai > 0
        try:
            m.setAvailableTorque(45.0)
        except AttributeError:
            pass
        m.setTorque(0.0)
        fl_motors.append(m)

    # ── FL sensors
    fl_sensors = []
    for sn in FL_SENSOR_NAMES:
        s = robot.getDevice(sn)
        s.enable(dt)
        fl_sensors.append(s)

    vel_est = VelocityEstimator(3, alpha=0.3)
    log, step = [], 0

    print("=" * 65)
    print("Spot SMC — FL Leg (dung dung cong thuc M/C/G)")
    print(f"  lambda={LAMBDA}  K={K}  phi={PHI}  limit={TORQUE_LIMIT} Nm")
    print(f"  Q_desired={Q_DESIRED}")
    print(f"  Dynamic model: compute_MCG() tu spot_dynamic_model.py")
    print("=" * 65)

    while robot.step(dt) != -1:
        t  = step * dt / 1000.0
        q  = np.array([s.getValue() for s in fl_sensors])
        dq = vel_est.update(q)

        # Tinh torque theo cong thuc SMC chinh xac
        tau = compute_smc_torque(q, dq)

        # Apply torque
        for i, m in enumerate(fl_motors):
            m.setTorque(float(tau[i]))

        # Log moi 10 steps (~320ms)
        if step % 10 == 0:
            e    = Q_DESIRED - q
            s_val = (DQ_DESIRED - dq) + LAMBDA * e
            M_diag = np.diag(
                compute_MCG(q, dq, HIP_POS, LEG_SIDE)[0]
            )
            print(
                f"t={t:6.2f}s | "
                f"q={np.round(q,3)} | "
                f"e_norm={np.linalg.norm(e):.4f} | "
                f"tau={np.round(tau,2)}"
            )
            print(
                f"         s={np.round(s_val,3)} | "
                f"M_diag={np.round(M_diag,4)}"
            )
            log.append({
                't':     t,
                'q':     q.tolist(),
                'e':     e.tolist(),
                's':     s_val.tolist(),
                'tau':   tau.tolist(),
                'M_diag': M_diag.tolist(),
            })

        step += 1

    # Save log
    path = os.path.join(os.path.dirname(__file__), 'smc_final_log.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n[SMC] Log saved -> {path}  ({len(log)} entries)")


if __name__ == '__main__':
    main()