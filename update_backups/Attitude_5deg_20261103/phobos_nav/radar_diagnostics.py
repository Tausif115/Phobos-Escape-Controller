import numpy as np


def format_radar_diagnostic(
    timestamp,
    shuttle_position,
    shuttle_velocity,
    raw_data,
    max_objects=3
):
    """Produce a compact diagnostic string without altering state."""

    shuttle_position = np.asarray(
        shuttle_position, dtype=float
    ).reshape(3)

    shuttle_velocity = np.asarray(
        shuttle_velocity, dtype=float
    ).reshape(3)

    lines = [
        f"t={timestamp:.2f}",
        f"shuttle_pos={np.round(shuttle_position, 2)}",
        f"shuttle_vel={np.round(shuttle_velocity, 2)}"
    ]

    for index, item in enumerate(raw_data[:max_objects]):
        matrix = np.asarray(item, dtype=float)

        if matrix.shape != (3, 3):
            continue

        raw_vector = matrix[0, :]
        absolute_candidate = raw_vector
        relative_candidate = shuttle_position + raw_vector

        lines.extend([
            f"debris[{index}] raw={np.round(raw_vector, 2)}",
            f"debris[{index}] norm={np.linalg.norm(raw_vector):.2f}",
            f"debris[{index}] absolute_candidate="
            f"{np.round(absolute_candidate, 2)}",
            f"debris[{index}] relative_candidate="
            f"{np.round(relative_candidate, 2)}",
            f"debris[{index}] closest={matrix[1, 0]:.2f}",
            f"debris[{index}] range_rate={matrix[1, 1]:.2f}",
            f"debris[{index}] diameter={matrix[1, 2]:.2f}"
        ])

    return " | ".join(lines)
