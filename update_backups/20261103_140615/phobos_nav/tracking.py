
from dataclasses import dataclass
import numpy as np

from .geometry import vec3, norm, unit


@dataclass
class DebrisTrack:
    identifier: int
    position: np.ndarray
    velocity: np.ndarray
    diameter: float
    timestamp: float
    hits: int = 1
    confirmed: bool = False

    def position_at(self, timestamp):
        dt = max(0.0, float(timestamp) - self.timestamp)
        return self.position + self.velocity * dt


class DebrisTracker:

    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.next_identifier = 1
        self.last_timestamp = None

    def reset(self):
        self.tracks.clear()
        self.next_identifier = 1
        self.last_timestamp = None

    def update(
        self,
        observations,
        timestamp,
        shuttle_position,
        shuttle_velocity
    ):
        cfg = self.config
        timestamp = float(timestamp)

        if not np.isfinite(timestamp):
            return self.tracks

        if (
            self.last_timestamp is not None
            and timestamp < self.last_timestamp
        ):
            self.reset()

        self.last_timestamp = timestamp

        shuttle_position = vec3(shuttle_position)
        shuttle_velocity = vec3(shuttle_velocity)

        # Keep only recently observed tracks.
        self.tracks = [
            track for track in self.tracks
            if 0.0 <= timestamp - track.timestamp <= cfg.track_max_age_s
        ]

        valid_observations = []

        for obs in observations:
            try:
                position = vec3(obs.position)
                diameter = float(obs.diameter)
                range_rate = float(obs.range_rate)

                if not np.all(np.isfinite(position)):
                    continue
                if not np.isfinite(diameter) or diameter < 0.0:
                    continue
                if not np.isfinite(range_rate):
                    continue

                valid_observations.append(obs)
            except (TypeError, ValueError, AttributeError):
                continue

        # Predict existing tracks.
        predicted_positions = [
            track.position_at(timestamp)
            for track in self.tracks
        ]

        candidate_pairs = []

        for ti, track in enumerate(self.tracks):
            for oi, obs in enumerate(valid_observations):
                distance = norm(
                    vec3(obs.position) - predicted_positions[ti]
                )

                # Diameter is useful as a weak association cue,
                # but should not be treated as a unique object ID.
                diameter_difference = abs(
                    float(obs.diameter) - track.diameter
                )

                diameter_tolerance = max(
                    1.0,
                    0.5 * max(float(obs.diameter), track.diameter)
                )

                if (
                    distance <= cfg.association_gate_m
                    and diameter_difference <= diameter_tolerance
                ):
                    cost = distance + 0.5 * diameter_difference
                    candidate_pairs.append((cost, ti, oi))

        candidate_pairs.sort(key=lambda x: x[0])

        assigned_tracks = set()
        assigned_observations = set()

        for _, ti, oi in candidate_pairs:
            if ti in assigned_tracks or oi in assigned_observations:
                continue

            track = self.tracks[ti]
            obs = valid_observations[oi]

            dt = timestamp - track.timestamp

            if dt <= 0.02:
                # Mark this pair as used, but do not estimate
                # velocity from a near-zero time interval.
                assigned_tracks.add(ti)
                assigned_observations.add(oi)
                continue

            predicted = predicted_positions[ti]
            residual = vec3(obs.position) - predicted

            track.position = predicted + cfg.alpha * residual
            track.velocity = (
                track.velocity + cfg.beta * residual / dt
            )
            track.diameter = float(obs.diameter)
            track.timestamp = timestamp
            track.hits += 1
            track.confirmed = track.hits >= 3

            assigned_tracks.add(ti)
            assigned_observations.add(oi)

        # Initialize unmatched detections as tentative tracks.
        for oi, obs in enumerate(valid_observations):
            if oi in assigned_observations:
                continue

            relative_position = (
                vec3(obs.position) - shuttle_position
            )
            radial_direction = unit(relative_position)

            # Radial-only initial estimate. Tangential motion
            # remains unknown until repeated observations.
            velocity_estimate = (
                shuttle_velocity
                + float(obs.range_rate) * radial_direction
            )

            self.tracks.append(
                DebrisTrack(
                    identifier=self.next_identifier,
                    position=vec3(obs.position).copy(),
                    velocity=velocity_estimate,
                    diameter=float(obs.diameter),
                    timestamp=timestamp,
                    hits=1,
                    confirmed=False
                )
            )

            self.next_identifier += 1

        return self.tracks
