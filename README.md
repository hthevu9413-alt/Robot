# Spot SMC Controller
Sliding Mode Control for Boston Dynamics Spot - Webots R2025a + ROS2 Jazzy

## Control Law
    e   = q_d - q
    s   = e_dot + lambda * e
    tau = M(q)*(ddq_d + lambda*e_dot) + C(q,dq)*dq + G(q) + K*sat(s/phi)

## Structure
- spot_smc_controller/spot_dynamic_model.py   : Task 3 - M/C/G dynamic model
- spot_smc_controller/spot_smc_one_leg_final.py : Task 4 - SMC 1 leg
- spot_smc_controller/spot_smc_4legs.py       : Task 4 - SMC 4 legs
- resource/spot.urdf                          : Robot URDF
- worlds/spot_ros2.wbt                        : Webots world
- results/smc_results.png                     : Simulation results
