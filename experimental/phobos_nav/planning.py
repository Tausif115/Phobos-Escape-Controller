
import numpy as np

from .geometry import (
    unit,
    norm,
    limit_norm,
    perpendicular_basis,
    swept_separation
)


def beeline_guidance(
    position,
    velocity,
    target_position,
    config
):
    direction = unit(target_position - position)

    desired_velocity = (
        config.goal_speed * direction
    )

    return config.velocity_gain * (
        desired_velocity - velocity
    )


class PredictivePlanner:

    def __init__(self, config):
        self.config = config
        self.last_choice = "beeline"

    def safety_radius(self, track, future_time):
        cfg = self.config

        # New tracks have more uncertain velocity.
        extra_new_track_uncertainty = (
            3.0 if track.hits < 2 else 0.0
        )

        return (
            cfg.shuttle_radius_m
            + 0.5 * track.diameter
            + cfg.clearance_margin_m
            + cfg.initial_uncertainty_m
            + cfg.uncertainty_growth_mps * future_time
            + extra_new_track_uncertainty
        )

    def candidate_velocities(self, goal_direction):
        cfg = self.config

        side1, side2 = perpendicular_basis(
            goal_direction
        )

        candidates = [
            (
                "beeline",
                cfg.goal_speed * goal_direction
            )
        ]

        for offset in (0.4, 0.8, 1.4):
            directions = [
                ("side1+", goal_direction + offset * side1),
                ("side1-", goal_direction - offset * side1),
                ("side2+", goal_direction + offset * side2),
                ("side2-", goal_direction - offset * side2)
            ]

            for name, direction in directions:
                candidates.append((
                    f"{name}{offset}",
                    cfg.goal_speed * unit(direction)
                ))

        # Braking can be safer than steering around
        # an object that crosses directly ahead.
        candidates.append((
            "brake",
            np.zeros(3)
        ))

        return candidates

    def rollout(
        self,
        position,
        velocity,
        target_position,
        target_velocity,
        tracks,
        timestamp,
        desired_velocity
    ):
        cfg = self.config

        p = position.copy()
        v = velocity.copy()

        risk_cost = 0.0
        effort_cost = 0.0
        minimum_clearance = float("inf")

        steps = max(
            1,
            int(np.ceil(cfg.horizon_s / cfg.step_s))
        )

        dt = cfg.horizon_s / steps

        for step in range(steps):
            tau0 = step * dt
            tau1 = (step + 1) * dt

            acceleration = cfg.velocity_gain * (
                desired_velocity - v
            )

            # This is a prediction-model constraint,
            # not a verified simulator thrust limit.
            acceleration = limit_norm(
                acceleration,
                cfg.predicted_max_accel
            )

            next_position = (
                p + v * dt
                + 0.5 * acceleration * dt * dt
            )

            next_velocity = (
                v + acceleration * dt
            )

            effort_cost += norm(acceleration) * dt

            for track in tracks:
                debris_start = track.position_at(
                    timestamp + tau0
                )

                debris_end = track.position_at(
                    timestamp + tau1
                )

                separation, fraction = swept_separation(
                    p,
                    next_position,
                    debris_start,
                    debris_end
                )

                time_at_closest = (
                    tau0 + fraction * dt
                )

                radius = self.safety_radius(
                    track,
                    time_at_closest
                )

                clearance = separation - radius

                minimum_clearance = min(
                    minimum_clearance,
                    clearance
                )

                if clearance < 0.0:
                    risk_cost += (
                        2000.0 * clearance * clearance
                    )

                elif clearance < cfg.replan_buffer_m:
                    remaining = (
                        cfg.replan_buffer_m - clearance
                    )

                    risk_cost += (
                        20.0 * remaining * remaining
                    )

            p = next_position
            v = next_velocity

        predicted_target = (
            target_position
            + target_velocity * cfg.horizon_s
        )

        terminal_distance = norm(
            predicted_target - p
        )

        return {
            "risk": risk_cost,
            "effort": effort_cost,
            "distance": terminal_distance,
            "clearance": minimum_clearance
        }

    def plan(
        self,
        position,
        velocity,
        target_position,
        target_velocity,
        tracks,
        timestamp
    ):
        cfg = self.config

        goal_direction = unit(
            target_position - position
        )

        candidates = self.candidate_velocities(
            goal_direction
        )

        evaluated = []

        for name, desired_velocity in candidates:
            metrics = self.rollout(
                position,
                velocity,
                target_position,
                target_velocity,
                tracks,
                timestamp,
                desired_velocity
            )

            if norm(desired_velocity) > 1.0e-9:
                direction = unit(desired_velocity)

                deviation = (
                    1.0
                    - float(np.dot(
                        direction,
                        goal_direction
                    ))
                )
            else:
                deviation = 1.0

            cost = (
                metrics["risk"]
                + 2.0 * metrics["distance"]
                + 0.1 * metrics["effort"]
                + 10.0 * deviation
            )

            evaluated.append((
                cost,
                name,
                desired_velocity,
                metrics
            ))

        nominal = evaluated[0]

        # If the original path is predicted clear,
        # do not modify the proven controller.
        if (
            nominal[3]["clearance"]
            >= cfg.replan_buffer_m
        ):
            self.last_choice = "beeline"

            return (
                beeline_guidance(
                    position,
                    velocity,
                    target_position,
                    cfg
                ),
                nominal[3]
            )

        feasible = [
            candidate for candidate in evaluated
            if candidate[3]["clearance"] >= 0.0
        ]

        if feasible:
            chosen = min(
                feasible,
                key=lambda candidate: candidate[0]
            )
        else:
            # Emergency heuristic:
            # select greatest predicted clearance.
            # NOT a guaranteed collision-free action.
            chosen = max(
                evaluated,
                key=lambda candidate:
                candidate[3]["clearance"]
            )

        _, name, desired_velocity, metrics = chosen

        self.last_choice = name

        acceleration = cfg.velocity_gain * (
            desired_velocity - velocity
        )

        return acceleration, metrics
