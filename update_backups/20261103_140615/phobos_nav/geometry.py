
import numpy as np

EPS = 1.0e-9


def vec3(value):
    v = np.asarray(value, dtype=float).reshape(3)

    if not np.all(np.isfinite(v)):
        raise ValueError("Nonfinite 3D vector")

    return v


def norm(value):
    return float(np.linalg.norm(value))


def unit(value, fallback=None):
    v = vec3(value)
    length = norm(v)

    if length > EPS:
        return v / length

    if fallback is not None:
        return unit(fallback)

    return np.array([1.0, 0.0, 0.0])


def limit_norm(value, maximum):
    v = vec3(value)
    length = norm(v)

    if maximum > 0.0 and length > maximum:
        return v * (maximum / length)

    return v


def perpendicular_basis(direction):
    forward = unit(direction)

    reference = np.array([0.0, -1.0, 0.0])

    if abs(float(np.dot(forward, reference))) > 0.95:
        reference = np.array([0.0, 0.0, 1.0])

    side1 = unit(np.cross(forward, reference))
    side2 = unit(np.cross(forward, side1))

    return side1, side2


def closest_approach(
    position_a,
    velocity_a,
    position_b,
    velocity_b,
    horizon_s
):
    """CPA for two constant-velocity objects.

    Returns:
        time_of_cpa, distance_at_cpa
    """
    relative_position = (
        vec3(position_b) - vec3(position_a)
    )

    relative_velocity = (
        vec3(velocity_b) - vec3(velocity_a)
    )

    speed_squared = float(
        np.dot(relative_velocity, relative_velocity)
    )

    if speed_squared < EPS:
        time_cpa = 0.0
    else:
        time_cpa = -float(
            np.dot(relative_position, relative_velocity)
        ) / speed_squared

    time_cpa = float(
        np.clip(time_cpa, 0.0, max(0.0, horizon_s))
    )

    separation = (
        relative_position
        + relative_velocity * time_cpa
    )

    return time_cpa, norm(separation)


def swept_separation(
    shuttle_start,
    shuttle_end,
    debris_start,
    debris_end
):
    """Minimum separation of linearly interpolated positions.

    Both objects move over the SAME time interval.
    This checks between trajectory sample endpoints.
    """
    relative_start = (
        vec3(shuttle_start) - vec3(debris_start)
    )

    relative_end = (
        vec3(shuttle_end) - vec3(debris_end)
    )

    delta = relative_end - relative_start

    denominator = float(np.dot(delta, delta))

    if denominator < EPS:
        fraction = 0.0
    else:
        fraction = float(np.clip(
            -np.dot(relative_start, delta)
            / denominator,
            0.0,
            1.0
        ))

    closest = relative_start + fraction * delta

    return norm(closest), fraction
