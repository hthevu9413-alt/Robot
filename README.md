# Spot SMC Controller

Sliding Mode Control (SMC) for Boston Dynamics Spot quadruped robot using Webots R2025a + ROS2 Jazzy.

## Project Structure

```
spot_smc_controller/
├── README.md
├── spot_smc_controller/
│   ├── spot_dynamic_model.py       # Task 3: Dynamic model M(q), C(q,dq), G(q)
│   ├── spot_smc_one_leg_final.py   # Task 4: SMC controller - 1 leg (FL)
│   ├── spot_smc_4legs.py           # Task 4: SMC controller - 4 legs
│   └── spot_driver.py              # Task 2: ROS2 driver
├── resource/
│   └── spot.urdf                   # Spot robot URDF model
├── worlds/
│   └── spot_ros2.wbt               # Webots world file
└── results/
    └── smc_results.png             # Simulation tracking error plot
```

## Requirements

- Ubuntu 22.04
- ROS2 Jazzy
- Webots R2025a
- Python 3.10+
- numpy

## Setup

```bash
# Clone repo
git clone https://github.com/hthevu9413-alt/Robot.git
cd Robot

# Build ROS2 package
cd ~/ros2_ws
colcon build --packages-select spot_smc_controller
source install/setup.bash
```

## Running

```bash
# 1. Open Webots and load world
# worlds/spot_ros2.wbt

# 2. Run SMC controller (1 leg)
cd spot_smc_controller
python3 spot_smc_one_leg_final.py

# 3. Run SMC controller (4 legs)
python3 spot_smc_4legs.py
```

## Control Law (SMC)

```
e     = q_d - q
s     = e_dot + lambda * e
tau   = M(q) * (ddq_d + lambda * e_dot) + C(q,dq) * dq + G(q) + K * sat(s/phi)
```

- `M(q)` — Inertia matrix (Euler-Lagrange, includes rotational inertia)
- `C(q,dq)` — Coriolis-centrifugal matrix (Christoffel symbols)
- `G(q)` — Gravity torque vector
- `K * sat(s/phi)` — Switching term with boundary layer (chattering reduction)

## Parameters

| Parameter | Value | Description |
|---|---|---|
| lambda | 3.0 | Sliding surface slope |
| K | 15.0 | Switching gain |
| phi | 0.15 | Boundary layer width |
| TORQUE_LIMIT | 40.0 Nm | Motor torque limit |

## Dynamic Model (Task 3)

Parameters extracted from `SpotLeg.proto` (Webots R2025a):

| Link | Mass (kg) | Notes |
|---|---|---|
| Shoulder | 2.132 | Hip abduction |
| Upper arm | 0.935 | Thigh |
| Forearm | 0.137 | Shank |
| Body | 14.410 | Main body |
| **Total** | **27.23** | 4 legs + body |

## Known Limitations

- Dynamic model uses block-diagonal assumption (legs independent)
- FK has y-axis offset due to URDF exported from simulation pose
- Robot unstable when lifting single leg (CoM shifts)
- Webots internal joint damping not modeled in M/C/G

## Team

HCMUT - Control Engineering Project 2025
