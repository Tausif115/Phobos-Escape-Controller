
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.geometry import (
    closest_approach,
    swept_separation,
    perpendicular_basis
)


class TestGeometry(unittest.TestCase):

    def test_head_on_collision(self):
        t, distance = closest_approach(
            [0, 0, 0],
            [1, 0, 0],
            [10, 0, 0],
            [-1, 0, 0],
            10.0
        )

        self.assertAlmostEqual(t, 5.0)
        self.assertAlmostEqual(distance, 0.0)

    def test_parallel_motion(self):
        t, distance = closest_approach(
            [0, 0, 0],
            [1, 0, 0],
            [0, 5, 0],
            [1, 0, 0],
            10.0
        )

        self.assertAlmostEqual(t, 0.0)
        self.assertAlmostEqual(distance, 5.0)

    def test_between_sample_collision(self):
        distance, fraction = swept_separation(
            [0, 0, 0],
            [10, 0, 0],
            [5, 5, 0],
            [5, -5, 0]
        )

        self.assertAlmostEqual(distance, 0.0)
        self.assertAlmostEqual(fraction, 0.5)

    def test_perpendicular_basis(self):
        direction = np.array([1., 0., 0.])

        side1, side2 = perpendicular_basis(
            direction
        )

        self.assertAlmostEqual(
            np.dot(direction, side1), 0.0
        )

        self.assertAlmostEqual(
            np.dot(direction, side2), 0.0
        )

        self.assertAlmostEqual(
            np.dot(side1, side2), 0.0
        )


if __name__ == "__main__":
    unittest.main()
