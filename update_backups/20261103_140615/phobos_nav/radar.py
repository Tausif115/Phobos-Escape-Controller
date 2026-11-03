
from dataclasses import dataclass
import numpy as np

from .geometry import vec3, norm


@dataclass
class RadarObservation:
    position: np.ndarray
    diameter: float
    range_rate: float


def decode_radar(raw_data, shuttle_position):
    """Decode official 3x3 debris matrices.

    The returned positions are treated as absolute
    Phobos-frame coordinates under the provisional
    radar-frame assumption.
    """
    shuttle_position = vec3(shuttle_position)
    observations = []

    for item in raw_data:
        try:
            matrix = np.asarray(item, dtype=float)

            if matrix.shape != (3, 3):
                continue

            relative_position = vec3(matrix[0, :])

            diameter = float(matrix[1, 2])
            range_rate = float(matrix[1, 1])

            if norm(relative_position) < 1.0e-9:
                continue

            if not np.isfinite(diameter):
                continue

            if diameter < 0.0:
                continue

            if not np.isfinite(range_rate):
                range_rate = 0.0

            observations.append(
                RadarObservation(
                    position=(
                        shuttle_position + relative_position
                    ),
                    diameter=diameter,
                    range_rate=range_rate
                )
            )

        except (ValueError, TypeError, IndexError):
            continue

    return observations
