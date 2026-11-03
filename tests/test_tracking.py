
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimental"))

from phobos_nav.config import CONFIG
from phobos_nav.radar import RadarObservation
from phobos_nav.tracking import DebrisTracker


class TestTracking(unittest.TestCase):

    def test_single_object_velocity_estimate(self):
        tracker = DebrisTracker(CONFIG)
        zero = np.zeros(3)

        tracker.update(
            [
                RadarObservation(
                    np.array([10., 0., 0.]),
                    1.0,
                    0.0
                )
            ],
            0.0,
            zero,
            zero
        )

        tracker.update(
            [
                RadarObservation(
                    np.array([11., 0., 0.]),
                    1.0,
                    0.0
                )
            ],
            1.0,
            zero,
            zero
        )

        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].hits, 2)

        self.assertGreater(
            tracker.tracks[0].velocity[0],
            0.0
        )

    def test_stale_track_removal(self):
        tracker = DebrisTracker(CONFIG)
        zero = np.zeros(3)

        tracker.update(
            [
                RadarObservation(
                    np.array([10., 0., 0.]),
                    1.0,
                    0.0
                )
            ],
            0.0,
            zero,
            zero
        )

        tracker.update(
            [],
            CONFIG.track_max_age_s + 1.0,
            zero,
            zero
        )

        self.assertEqual(len(tracker.tracks), 0)


if __name__ == "__main__":
    unittest.main()
