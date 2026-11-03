
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.config import CONFIG
from phobos_nav.tracking import DebrisTrack
from phobos_nav.planning import (
    PredictivePlanner,
    beeline_guidance
)


class TestPlanning(unittest.TestCase):

    def setUp(self):
        self.planner = PredictivePlanner(CONFIG)

        self.position = np.array([0., 0., 0.])
        self.velocity = np.array([0., 0., 0.])

        self.target = np.array([100., 0., 0.])
        self.target_velocity = np.zeros(3)

    def test_nominal_guidance(self):
        acceleration = beeline_guidance(
            self.position,
            self.velocity,
            self.target,
            CONFIG
        )

        self.assertAlmostEqual(
            acceleration[0],
            CONFIG.goal_speed * CONFIG.velocity_gain
        )

    def test_no_obstacles(self):
        acceleration, metrics = self.planner.plan(
            self.position,
            self.velocity,
            self.target,
            self.target_velocity,
            [],
            0.0
        )

        self.assertEqual(
            self.planner.last_choice,
            "beeline"
        )

        self.assertTrue(
            np.all(np.isfinite(acceleration))
        )

    def test_obstacle_ahead_triggers_replanning(self):
        obstacle = DebrisTrack(
            identifier=1,
            position=np.array([10., 0., 0.]),
            velocity=np.zeros(3),
            diameter=2.0,
            timestamp=0.0,
            hits=3
        )

        acceleration, metrics = self.planner.plan(
            self.position,
            self.velocity,
            self.target,
            self.target_velocity,
            [obstacle],
            0.0
        )

        self.assertNotEqual(
            self.planner.last_choice,
            "beeline"
        )

        self.assertTrue(
            np.all(np.isfinite(acceleration))
        )


if __name__ == "__main__":
    unittest.main()
