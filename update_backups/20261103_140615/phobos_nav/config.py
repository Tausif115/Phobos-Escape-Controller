
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # BASELINE: original translation controller
    # MONITOR:  radar/tracking without avoidance
    # AVOID:    enable experimental avoidance
    mode: str = "MONITOR"

    # Baseline guidance
    goal_speed: float = 10.0
    velocity_gain: float = 1.0

    # Baseline attitude control
    angular_velocity_gain: float = 0.2
    kp_roll: float = 2.0
    kp_pitch: float = 2.0
    kp_yaw: float = 2.0

    # Radar and tracking
    radar_max_objects: int = 16
    track_max_age_s: float = 2.0
    association_gate_m: float = 20.0
    alpha: float = 0.65
    beta: float = 0.20

    # Predictive planning
    horizon_s: float = 8.0
    step_s: float = 0.5

    # Model only: NOT verified actuator limits
    predicted_max_accel: float = 2.0

    # Provisional geometry and uncertainty estimates
    shuttle_radius_m: float = 2.0
    clearance_margin_m: float = 4.0
    initial_uncertainty_m: float = 2.0
    uncertainty_growth_mps: float = 0.5

    # Replan when nominal predicted clearance is
    # below this buffer.
    replan_buffer_m: float = 5.0

    # Diagnostic logging
    log_every_n: int = 40


CONFIG = Config()
