
import numpy as np

from .geometry import unit


def angular_velocity_from_quaternions(q1, q2, dt):
    """Match the official starter controller convention."""
    q1 = np.asarray(q1, dtype=float).reshape(4)
    q2 = np.asarray(q2, dtype=float).reshape(4)

    # q and -q represent the same orientation.
    if np.dot(q1, q2) < 0.0:
        q2 = -q2

    return (2.0 / dt) * np.array([
        q1[3]*q2[0] - q1[0]*q2[3]
        - q1[1]*q2[2] + q1[2]*q2[1],

        q1[3]*q2[1] + q1[0]*q2[2]
        - q1[1]*q2[3] - q1[2]*q2[0],

        q1[3]*q2[2] - q1[0]*q2[1]
        + q1[1]*q2[0] - q1[2]*q2[3]
    ])


def attitude_acceleration(
    st,
    shuttle,
    shuttle_frame,
    phobos_frame,
    current_quaternion,
    goal_direction,
    config
):
    x_axis = unit(goal_direction)

    reference = np.array([0.0, -1.0, 0.0])

    if abs(float(np.dot(x_axis, reference))) > 0.95:
        reference = np.array([0.0, 0.0, 1.0])

    z_axis = unit(np.cross(x_axis, reference))
    y_axis = unit(np.cross(z_axis, x_axis))

    target_dcm = np.vstack((
        x_axis,
        y_axis,
        z_axis
    ))

    target_quaternion = st.math.DCM_to_Quat(
        target_dcm
    )

    desired_rate = angular_velocity_from_quaternions(
        current_quaternion,
        target_quaternion,
        1.0 / config.angular_velocity_gain
    )

    desired_rate = st.frames.FramedAngVel(
        desired_rate,
        phobos_frame
    ).WRT(phobos_frame).ExprIn(shuttle_frame)

    current_rate = (
        shuttle.getAngVelocity()
        .WRT(phobos_frame)
        .ExprIn(shuttle_frame)
    )

    error = (
        np.asarray(desired_rate, dtype=float)
        - np.asarray(current_rate, dtype=float)
    )

    torque_strength = shuttle.GetParam(
        st.VarType.double,
        ["Thrusters", "TorqueStrength_Nm"]
    )

    commanded_torque = torque_strength * np.array([
        config.kp_roll * error[0],
        config.kp_pitch * error[1],
        config.kp_yaw * error[2]
    ])

    inertia = np.asarray(
        shuttle.getInertiaTensor(shuttle_frame),
        dtype=float
    )

    return np.linalg.solve(
        inertia,
        commanded_torque
    )
