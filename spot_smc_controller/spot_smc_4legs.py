"""
spot_smc_4legs_final.py
=======================
SMC controller cho toan bo 4 chan Spot (12 DOF).
Dung DUNG cong thuc theo yeu cau de tai:

    tau = M(q)*(ddq_d + lambda*e_dot) + C(q,dq)*dq + G(q) + K*sat(s/phi)
    s   = e_dot + lambda*e
    e   = q_d - q

Dynamic model: spot_dynamic_model_v2.py (Task 3, v2 - day du rotational inertia).
Moi chan co hip_pos va side rieng (FL/RL = 'L', FR/RR = 'R').
4 chan dong thoi dung SMC torque control, huong ve standing pose.

Chay:
    Dat file nay cung thu muc voi spot_dynamic_model_v2.py
    Webots phai dang chay world spot_ros2.wbt voi controller="<extern>"
    python3 spot_smc_4legs_final.py
"""

from controller import Robot
import numpy as np
import os, json, sys

# Import dynamic model v2 (cung thu muc)
sys.path.insert(0, os.path.dirname(__file__))
from spot_dynamic_model_v2 import compute_MCG, HIP_POSITIONS

# ══════════════════════════════════════════════════════════════════
# SMC PARAMETERS
# ══════════════════════════════════════════════════════════════════
# Dua tren ket qua 1 chan (K=5 da kha on dinh):
#   - LAMBDA = 2.0  : toc do hoi tu sliding surface
#   - K      = 5.0  : switching gain (du de bu gravity ~3-6 Nm)
#   - PHI    = 0.1  : boundary layer (giam chattering)
#
# M diagonal v2 (voi rotational inertia): ~[0.15-0.30, 0.10-0.15, 0.01-0.02]
# => tau_nominal = M * (ddq_d + lambda*e_dot) lon hon v1
# => K=5 la reasonable, co the tang len 8-10 neu can

LAMBDA       = 8.0
K            = 15.0
PHI          = 0.15
TORQUE_LIMIT = 40.0
TIMESTEP     = 32      # ms

# Target: standing pose
# q = [abduction, rotation, elbow]
# [0.0, 0.7, 0.0] la tu the dung on dinh trong Webots
Q_DESIRED   = np.array([0.0,  0.7,  0])
DQ_DESIRED  = np.zeros(3)
DDQ_DESIRED = np.zeros(3)

# ══════════════════════════════════════════════════════════════════
# LEG DEFINITIONS
# Moi chan: (ten, side, hip_pos, motor_names, sensor_names)
# ══════════════════════════════════════════════════════════════════

ALL_LEGS = [
    (
        'FL', 'L', HIP_POSITIONS['FL'],
        ['front left shoulder abduction motor',
         'front left shoulder rotation motor',
         'front left elbow motor'],
        ['front left shoulder abduction sensor',
         'front left shoulder rotation sensor',
         'front left elbow sensor'],
    ),
    (
        'FR', 'R', HIP_POSITIONS['FR'],
        ['front right shoulder abduction motor',
         'front right shoulder rotation motor',
         'front right elbow motor'],
        ['front right shoulder abduction sensor',
         'front right shoulder rotation sensor',
         'front right elbow sensor'],
    ),
    (
        'RL', 'L', HIP_POSITIONS['RL'],
        ['rear left shoulder abduction motor',
         'rear left shoulder rotation motor',
         'rear left elbow motor'],
        ['rear left shoulder abduction sensor',
         'rear left shoulder rotation sensor',
         'rear left elbow sensor'],
    ),
    (
        'RR', 'R', HIP_POSITIONS['RR'],
        ['rear right shoulder abduction motor',
         'rear right shoulder rotation motor',
         'rear right elbow motor'],
        ['rear right shoulder abduction sensor',
         'rear right shoulder rotation sensor',
         'rear right elbow sensor'],
    ),
]

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def sat(x, phi):
    """Saturation function thay sign() — giam chattering."""
    return np.clip(x / phi, -1.0, 1.0)


class VelocityEstimator:
    """Numerical differentiation + low-pass filter de uoc tinh dq tu q."""
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
# SMC TORQUE — DUNG DUNG CONG THUC CHINH XAC
# ══════════════════════════════════════════════════════════════════

def compute_smc_torque(q, dq, hip_pos, side):
    """
    Tinh torque SMC cho mot chan 3-DOF.

    Cong thuc:
        e      = q_d - q
        e_dot  = dq_d - dq
        s      = e_dot + lambda * e
        tau    = M(q)*(ddq_d + lambda*e_dot) + C(q,dq)*dq + G(q) + K*sat(s/phi)

    M, C, G lay tu spot_dynamic_model_v2.py (day du rotational inertia).
    hip_pos va side khac nhau cho tung chan.
    """
    M, C, G = compute_MCG(q, dq, hip_pos, side)

    e     = Q_DESIRED  - q
    e_dot = DQ_DESIRED - dq

    s = e_dot + LAMBDA * e

    # Nominal: bam sat dong luc hoc
    tau_nominal  = M @ (DDQ_DESIRED + LAMBDA * e_dot) + C @ dq + G

    # Switching: bu disturbance va uncertainty
    tau_switching = K * sat(s, PHI)

    tau = tau_nominal + tau_switching
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT), M, s


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    robot = Robot()
    dt    = TIMESTEP

    # ── Init motors, sensors, velocity estimators cho 4 chan
    all_motors  = []
    all_sensors = []
    all_vel_est = []

    for leg_name, side, hip_pos, mnames, snames in ALL_LEGS:
        leg_motors, leg_sensors = [], []

        for mn in mnames:
            m = robot.getDevice(mn)
            m.setPosition(float('inf'))   # disable position control
            m.setVelocity(10.0)           # phai > 0 de torque mode hoat dong
            try:
                m.setAvailableTorque(45.0)
            except AttributeError:
                pass
            m.setTorque(0.0)
            leg_motors.append(m)

        for sn in snames:
            s = robot.getDevice(sn)
            s.enable(dt)
            leg_sensors.append(s)

        all_motors.append(leg_motors)
        all_sensors.append(leg_sensors)
        all_vel_est.append(VelocityEstimator(3, alpha=0.3))

    print("=" * 70)
    print("Spot SMC 4-Leg Controller — Dung dung cong thuc M/C/G (v2)")
    print(f"  lambda={LAMBDA}  K={K}  phi={PHI}  torque_limit={TORQUE_LIMIT} Nm")
    print(f"  Q_desired={Q_DESIRED}")
    print(f"  Dynamic model: spot_dynamic_model_v2.py (voi rotational inertia)")
    print("=" * 70)

    log, step = [], 0

    while robot.step(dt) != -1:
        t = step * dt / 1000.0

        q_all   = []
        tau_all = []
        M_diags = []
        s_all   = []

        # ── Tinh va apply torque cho tung chan
        for li, (leg_name, side, hip_pos, _, _) in enumerate(ALL_LEGS):
            q   = np.array([s.getValue() for s in all_sensors[li]])
            dq  = all_vel_est[li].update(q)

            tau, M, s_val = compute_smc_torque(q, dq, hip_pos, side)

            for i, m in enumerate(all_motors[li]):
                m.setTorque(float(tau[i]))

            q_all.append(q)
            tau_all.append(tau)
            M_diags.append(np.diag(M))
            s_all.append(s_val)

        # ── Log moi 10 steps (~320ms)
        if step % 10 == 0:
            e_norms = [np.linalg.norm(Q_DESIRED - q_all[li]) for li in range(4)]
            e_mean  = np.mean(e_norms)
            leg_names = [leg[0] for leg in ALL_LEGS]

            print(
                f"t={t:6.2f}s | "
                f"e_norm [FL={e_norms[0]:.3f} FR={e_norms[1]:.3f} "
                f"RL={e_norms[2]:.3f} RR={e_norms[3]:.3f}] mean={e_mean:.3f}"
            )
            print(
                f"         FL: q={np.round(q_all[0],3)} "
                f"tau={np.round(tau_all[0],2)} "
                f"M_diag={np.round(M_diags[0],3)}"
            )
            print(
                f"         FR: q={np.round(q_all[1],3)} "
                f"tau={np.round(tau_all[1],2)}"
            )

            log.append({
                't':       t,
                'e_norms': e_norms,
                'e_mean':  e_mean,
                'q':       [q.tolist() for q in q_all],
                'tau':     [tau.tolist() for tau in tau_all],
                'M_diags': [md.tolist() for md in M_diags],
                's':       [sv.tolist() for sv in s_all],
            })

        step += 1

    # ── Save log
    path = os.path.join(os.path.dirname(__file__), 'smc_4legs_final_log.json')
    with open(path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n[SMC] Log saved -> {path}  ({len(log)} entries)")


if __name__ == '__main__':
    main()
