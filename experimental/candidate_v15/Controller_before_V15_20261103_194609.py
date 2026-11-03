#############################################################################################
# DON'T CHANGE ANY OF THE BELOW. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################

import os, sys, time, datetime, traceback
import spaceteams as st
def custom_exception_handler(exctype, value, tb):
    error_message = "".join(traceback.format_exception(exctype, value, tb))
    st.logger_fatal(error_message)
sys.excepthook = custom_exception_handler
st.connect_to_sim(sys.argv)

import numpy as np
import numpy.typing as npt


this = st.GetThisSystem()

doSpawning = st.SimGlobals.GetSimEntity().GetParam(st.VarType.bool, "DoSpawning")
if doSpawning:
    while True:
        time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

phobos: st.Entity = this.GetParam(st.VarType.entityRef, "Phobos")
phobosFrame = phobos.GetBodyFixedFrame()
shuttle: st.Entity = this.GetParam(st.VarType.entityRef, "Shuttle")
shuttleFrame = shuttle.GetBodyFixedFrame()
target: st.Entity = this.GetParam(st.VarType.entityRef, "Target")

topCamera: st.Entity = this.GetParam(st.VarType.entityRef, "TopCamera")
topCameraFrame = topCamera.GetBodyFixedFrame()
bottomCamera: st.Entity = this.GetParam(st.VarType.entityRef, "BottomCamera")
bottomCameraFrame = bottomCamera.GetBodyFixedFrame()


def GetNearestDebrisLocations(N: int) -> list[npt.NDArray]:
    allDebrisInfo: list[npt.NDArray] = shuttle.GetParamArray(st.VarType.doubleM3x3, 
                                                            ["Radar", "SortedDebrisData"])
    nearest = allDebrisInfo[:min(N, len(allDebrisInfo))]
    return nearest


def PanTiltCamera(camera: st.Entity, yaw_deg: float, pitch_deg: float) -> None:
    if camera == topCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, -50.0, 0.0)
    elif camera == bottomCamera:
        yaw_deg = np.clip(yaw_deg, -89.0, 89.0)
        pitch_deg = np.clip(pitch_deg, 0.0, 50.0)
    camera.setRotation_DCM(st.math.Euler321_to_DCM([np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), 0.0]), shuttleFrame)


def SetCommand(accelCommand: npt.NDArray, angAccelCommand: npt.NDArray) -> None:
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAccel"], accelCommand)
    shuttle.SetParam(st.VarType.doubleV3, ["JetSelection", "CommandedAngAccel"], angAccelCommand)


#############################################################################################
# DON'T CHANGE ANY OF THE ABOVE. NECESSARY FOR CORRECTLY JOINING THE COMPETITION SIMULATION #
#############################################################################################


#########################################################
# EXAMPLE USER CODE (NOTE: RUNNING A LOOP IS NECESSARY) #
#########################################################


import copy
from PIL import Image
from pathlib import Path


"""
Get angular velocity from two quaternions representing the orientation of 
an object at time (t0) and (t0 + dt). Assumes scalar-last quaternions.
"""
def AngVelFromQuats(q1: npt.NDArray, q2: npt.NDArray, dt: float):
    return (2.0 / dt) * np.array([
        q1[3]*q2[0] - q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1],
        q1[3]*q2[1] + q1[0]*q2[2] - q1[1]*q2[3] - q1[2]*q2[0],
        q1[3]*q2[2] - q1[0]*q2[1] + q1[1]*q2[0] - q1[2]*q2[3]])


from phobos_nav.config import CONFIG
from phobos_nav.radar import decode_radar
from phobos_nav.tracking import DebrisTracker
from phobos_nav.radar_diagnostics import format_radar_diagnostic

tracker = DebrisTracker(CONFIG)
###############################
# Get initial simulation time #
###############################

t0_sim: st.timestamp = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"])
t0: datetime.datetime = t0_sim.as_datetime()

##################
# Start the loop #
##################

iteration = 0
imageNumber = 0
exit_flag: bool = False
coast_mode = False
v13_pulse_end = -1.0
v13_next_decision = 0.0
v13_pulse_direction = np.zeros(3, dtype=float)
v13_pulse_count = 0
v13_last_phase = ""
while not exit_flag:

    ################################
    # Wait between loop iterations #
    ################################
    
    time.sleep(1.0 / this.GetParam(st.VarType.double, "LoopFreqHz"))

    ###################################
    # Get elapsed time from sim start #
    ###################################

    t_now: datetime.datetime = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    t_from_start: float = (t_now - t0).total_seconds()

    #####################################
    # Example for getting vehicle state #
    #####################################

    locVelAcc = shuttle.getLocVelAcc()
    loc = locVelAcc.loc_WRT_ExprIn(phobosFrame)
    vel = locVelAcc.vel_WRT_ExprIn(phobosFrame)
    rot = shuttle.getRotation().Quat_WRT(phobosFrame)
    angvel = shuttle.getAngVelocity().WRT_ExprIn(phobosFrame)

    #####################################################
    # Example for getting additional vehicle properties #
    #####################################################

    # mass = shuttle.getMass()
    # inertiaBodyFrame = shuttle.getInertiaTensor(shuttleFrame)
    # damage: float = shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])
    # fuelUsed: float = shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"])

    ##################################################################
    # Example for rotating the top camera 30 deg left and 10 deg up, #
    # and the bottom camera 20 deg right and 15 deg down.            #
    ##################################################################

    if iteration % 20 == 0:
        PanTiltCamera(topCamera, 30.0, -10.0)
        PanTiltCamera(bottomCamera, -20.0, 15.0)

    ####################################################################################
    # Example for obtaining an image from the top camera with FOV = 60 deg and EV = 12 #
    ####################################################################################

    if iteration % 20 == 0:  # Don't hot-loop the image captures
        cameraParams = st.CaptureImageProperties()
        cameraParams.nonphys_EV = 10.0
        cameraParams.FOV = 60.0
        cameraParams.ResolutionX = 512
        cameraParams.ResolutionY = 512
        image = st.camera.CaptureImageBlocking(topCamera, cameraParams)

        if image is not None:
            imageR = np.asarray(image.as_RGB8().PixelsR).reshape((cameraParams.ResolutionX, cameraParams.ResolutionY))
            imageG = np.asarray(image.as_RGB8().PixelsG).reshape((cameraParams.ResolutionX, cameraParams.ResolutionY))
            imageB = np.asarray(image.as_RGB8().PixelsB).reshape((cameraParams.ResolutionX, cameraParams.ResolutionY))
            imageRGB = Image.fromarray(np.dstack((imageR, imageG, imageB)).astype(np.uint8))

            # OPTIONAL: SAVE THE IMAGE TO DISK FOR VIEWING

            # savePath = os.path.join(st.path_utils.GetLocalOutputDir(), "STU4")
            # os.makedirs(savePath, exist_ok=True)
            # imageName = f"TestImage{str(imageNumber).zfill(4)}.png"
            # imageRGB.save(os.path.join(savePath, imageName))

            # st.OnScreenLogMessage(f"Saved camera capture to {os.path.join(savePath, imageName)}", "CameraCapture", st.Severity.Info)
            imageNumber += 1
        else:
            st.OnScreenLogMessage("Failed to capture camera image :(", "CameraNoCapture", st.Severity.Info)

    ###################################################################################
    # Example for getting the positions, closest points, range rates, and diameters   #
    # of the nearest 4 pieces of debris and printing the distance to the closest one. #
    ###################################################################################

    # debrisInfo = GetNearestDebrisLocations(4)
    # if len(debrisInfo) > 0:
    #     closestDebris = debrisInfo[0]
    #     closestDebris_loc = closestDebris[0, :]
    #     closestDebris_closestPoint = closestDebris[1, 0]
    #     closestDebris_rangeRate = closestDebris[1, 1]
    #     closestDebris_diameter = closestDebris[1, 2]
    #     st.OnScreenLogMessage(f"Nearest debris: {np.linalg.norm(closestDebris_loc):.1f} m", 
    #                           "UserAPIPrintout", st.Severity.Info)

    # ========================================================
    # EXPERIMENTAL RADAR MONITOR — READ ONLY
    # ========================================================

    if CONFIG.mode == "MONITOR":
        try:
            raw_debris = GetNearestDebrisLocations(
                CONFIG.radar_max_objects
            )

            observations = decode_radar(raw_debris, loc)

            if iteration % 100 == 0:
                diagnostic = format_radar_diagnostic(
                    t_from_start,
                    loc,
                    vel,
                    raw_debris,
                    max_objects=3
                )

                print("RADAR_DIAGNOSTIC:", diagnostic, flush=True)

            tracks = tracker.update(
                observations,
                t_from_start,
                loc,
                vel
            )

            if iteration % CONFIG.log_every_n == 0:
                st.OnScreenLogMessage(
                    f"MONITOR | Tracks={len(tracks)} "
                    f"Confirmed={sum(t.confirmed for t in tracks)} "
                    f"Radar={len(observations)}",
                    "PhobosNavigation",
                    st.Severity.Info
                )

        except Exception as error:
            if iteration % CONFIG.log_every_n == 0:
                st.OnScreenLogMessage(
                    f"Radar monitor error: {error}",
                    "PhobosNavigation",
                    st.Severity.Warning
                )
    ####################################################
    # Example for commanding a beeline acceleration    #
    # toward the target (and facing it the entire way) #
    ####################################################

    # ======================================================
    # CANDIDATE V2: TRANSLATION + ANGULAR VELOCITY DAMPING
    # ======================================================

    # Keep the proven translational velocity target.
    # Do not force the shuttle X-axis to face the target.

    targetState = target.getLocVelAcc()
    targetLoc = targetState.loc_WRT_ExprIn(phobosFrame)

    target_vector = np.asarray(targetLoc - loc, dtype=float)
    target_distance = float(np.linalg.norm(target_vector))

    if target_distance > 1e-9:
        goalDirection = target_vector / target_distance
    else:
        goalDirection = np.zeros(3)

    goalSpeed_m_s = 10.0
    goalVelocity = goalSpeed_m_s * goalDirection

    # =====================================================
    # V6 - EARLIER PREDICTIVE COASTING
    # =====================================================

    # Preserve the original V2 velocity controller.
    velocity_error = (
        np.asarray(goalVelocity, dtype=float)
        - np.asarray(vel, dtype=float)
    )

    accel_phobos = velocity_error.copy()

    # Maximum commanded acceleration from official config.
    max_linear_accel = 0.5

    accel_norm = float(np.linalg.norm(accel_phobos))

    if accel_norm > max_linear_accel:
        accel_phobos *= max_linear_accel / accel_norm

    # -----------------------------------------------------
    # Predict closest approach to the moving target.
    # Both position and velocity are in the Phobos frame.
    # -----------------------------------------------------

    target_velocity = np.asarray(
        targetState.vel_WRT_ExprIn(phobosFrame),
        dtype=float
    )

    relative_position = (
        np.asarray(targetLoc, dtype=float)
        - np.asarray(loc, dtype=float)
    )

    relative_velocity = (
        np.asarray(vel, dtype=float)
        - target_velocity
    )

    relative_speed_sq = float(
        np.dot(relative_velocity, relative_velocity)
    )

    time_to_cpa = float("inf")
    miss_distance = float("inf")

    if relative_speed_sq > 0.25:
        predicted_time = float(
            np.dot(relative_position, relative_velocity)
            / relative_speed_sq
        )

        if predicted_time > 0.0:
            time_to_cpa = predicted_time

            closest_position = (
                relative_position
                - relative_velocity * time_to_cpa
            )

            miss_distance = float(
                np.linalg.norm(closest_position)
            )

    # -----------------------------------------------------
    # Hysteresis prevents rapid coasting-mode switching.
    # Enter coast only on a clearly favorable trajectory.
    # -----------------------------------------------------

    previous_coast_mode = coast_mode

    if coast_mode:
        # Resume thrust if the prediction deteriorates.
        coast_mode = (
            1.0 < time_to_cpa < 105.0
            and miss_distance < 32.0
        )
    else:
        # Enter coast only with a conservative miss margin.
        coast_mode = (
            1.0 < time_to_cpa < 90.0
            and miss_distance < 18.0
        )

    if coast_mode:
        accel_phobos = np.zeros(3, dtype=float)

    # -----------------------------------------------------
    # Diagnostics: record fuel and coast decisions.
    # -----------------------------------------------------

    if iteration % 300 == 0 or coast_mode != previous_coast_mode:

        fuel_now = float(
            shuttle.GetParam(
                st.VarType.double,
                ["Stats", "FuelUsed"]
            )
        )

        distance_to_target = float(
            np.linalg.norm(relative_position)
        )

        print(
            f"FUEL_V8 "
            f"t={t_from_start:.2f} "
            f"fuel_kg={fuel_now:.3f} "
            f"coast={int(coast_mode)} "
            f"distance_m={distance_to_target:.2f} "
            f"t_cpa_s={time_to_cpa:.2f} "
            f"miss_m={miss_distance:.2f} "
            f"cmd_accel={np.linalg.norm(accel_phobos):.3f}",
            flush=True
        )

    # Transform the translation command into shuttle axes.
    linearAcceleration_m_s2 = (
        st.frames.FramedVector(accel_phobos, phobosFrame)
        .ExprIn(shuttleFrame)
    )

    # Angular velocity stabilization:
    # Do not turn toward the target continuously.
    current_AngVel = np.asarray(
        shuttle.getAngVelocity()
        .WRT(phobosFrame)
        .ExprIn(shuttleFrame),
        dtype=float
    )

    desired_AngVel = np.zeros(3)

    # Damping coefficient, units 1/s.
    damping_gain = 0.5

    angularAcceleration_rad_s2 = (
        -damping_gain * current_AngVel
    )

    # Conservative angular acceleration cap.
    max_angular_accel = 0.02
    angular_accel_norm = float(
        np.linalg.norm(angularAcceleration_rad_s2)
    )

    if angular_accel_norm > max_angular_accel:
        angularAcceleration_rad_s2 *= (
            max_angular_accel / angular_accel_norm
        )

    # Send commands through the official actuator interface.
    # ===========================================
    # V13: FUEL-AWARE RELATIVE VELOCITY GUIDANCE
    # ===========================================

    v13_fuel = float(shuttle.GetParam(
        st.VarType.double, ["Stats", "FuelUsed"]
    ))

    v13_r = (
        np.asarray(targetLoc, dtype=float)
        - np.asarray(loc, dtype=float)
    )

    v13_target_velocity = np.asarray(
        targetState.vel_WRT_ExprIn(phobosFrame),
        dtype=float
    )

    v13_v = (
        np.asarray(vel, dtype=float)
        - v13_target_velocity
    )

    v13_range = float(np.linalg.norm(v13_r))
    v13_rhat = v13_r / max(v13_range, 1e-9)

    # Positive means approaching the ship.
    v13_closing = float(
        np.dot(v13_v, v13_rhat)
    )

    # Velocity perpendicular to target direction.
    v13_lateral_v = (
        v13_v - v13_closing * v13_rhat
    )

    v13_lateral_speed = float(
        np.linalg.norm(v13_lateral_v)
    )

    # Adaptive closing-speed goal:
    # Faster when far away, slower near target.
    # V14: speed needed to reach the winning radius
    # against approximate observed deceleration.
    v13_decel_est = 0.0106
    v13_distance_to_win = max(v13_range - 50.0, 0.0)

    v13_goal_closing = float(
        np.clip(
            np.sqrt(
                2.0 * v13_decel_est * v13_distance_to_win
                + 0.8 ** 2
            ) + 0.20,
            0.8,
            6.0
        )
    )

    # Difference between desired and actual velocity.
    # Positive radial term: gain closing speed.
    # Lateral term: reduce sideways drift.
    v13_speed_deficit = max(
        v13_goal_closing - v13_closing, 0.0
    )

    v13_delta_v = (
        v13_speed_deficit * v13_rhat
        - 0.8 * v13_lateral_v
    )

    v13_error = float(
        np.linalg.norm(v13_delta_v)
    )

    # Reserve fuel for later stages of the approach.
    # V14: retain fuel for the later approach.
    if v13_range > 700.0:
        v13_allowed_fuel = 26.0
    elif v13_range > 250.0:
        v13_allowed_fuel = 30.0
    else:
        v13_allowed_fuel = 40.0

    v13_phase = "BOOST"

    if v13_fuel >= 21.0:
        # Switch from V8 initial boost to V13 guidance.
        v13_phase = "COAST"
        v13_command = np.zeros(3, dtype=float)

        # Continue a scheduled short correction burn.
        if (
            t_from_start < v13_pulse_end
            and v13_fuel < v13_allowed_fuel
        ):
            v13_phase = "CORRECT"
            v13_command = (
                0.5 * v13_pulse_direction
            )

        # Decide whether another correction is needed.
        elif (
            t_from_start >= v13_next_decision
            and v13_fuel < v13_allowed_fuel - 0.15
            and v13_range > 50.0
        ):
            v13_next_decision = t_from_start + 4.0

            if v13_error > 0.25:
                v13_pulse_direction = (
                    v13_delta_v / v13_error
                )

                v13_duration = float(
                    np.clip(
                        0.6 * v13_error / 0.5,
                        0.20,
                        1.0
                    )
                )

                v13_pulse_end = (
                    t_from_start + v13_duration
                )

                v13_command = (
                    0.5 * v13_pulse_direction
                )

                v13_pulse_count += 1
                v13_phase = "CORRECT"

        linearAcceleration_m_s2 = (
            st.frames.FramedVector(
                v13_command, phobosFrame
            ).ExprIn(shuttleFrame)
        )

    # Emergency spending limit.
    # This is NOT an exact propellant limit.
    if v13_fuel >= 40.0:
        linearAcceleration_m_s2 = np.zeros(3)
        angularAcceleration_rad_s2 = np.zeros(3)
        v13_phase = "BUDGET_LIMIT"

    if (
        iteration % 200 == 0
        or v13_phase != v13_last_phase
    ):
        print(
            f"FUEL_V14 "
            f"t={t_from_start:.2f} "
            f"phase={v13_phase} "
            f"fuel_kg={v13_fuel:.3f} "
            f"range_m={v13_range:.2f} "
            f"closing_m_s={v13_closing:.3f} "
            f"goal_m_s={v13_goal_closing:.3f} "
            f"lateral_m_s={v13_lateral_speed:.3f} "
            f"pulses={v13_pulse_count}",
            flush=True
        )

    v13_last_phase = v13_phase

    SetCommand(
        linearAcceleration_m_s2,
        angularAcceleration_rad_s2
    )


    # ==================================================
    # ATTITUDE DIAGNOSTICS - FIRST 30 SIMULATION SECONDS
    # READ ONLY - DOES NOT MODIFY THRUSTER COMMANDS
    # ==================================================

    if 0.0 <= t_from_start <= 30.0 and iteration % 20 == 0:
        print(
            f"ATTITUDE_DIAG "
            f"t={t_from_start:.3f} "
            f"ang_speed_deg_s="
            f"{np.degrees(np.linalg.norm(current_AngVel)):.4f} "
            f"desired_ang_speed_deg_s="
            f"{np.degrees(np.linalg.norm(desired_AngVel)):.4f} "
            f"angvel_body={np.round(current_AngVel, 5).tolist()} "
            f"cmd_ang_accel="
            f"{np.round(angularAcceleration_rad_s2, 5).tolist()} "
            f"cmd_lin_accel="
            f"{np.round(linearAcceleration_m_s2, 5).tolist()} "
            f"speed_mps={np.linalg.norm(vel):.4f}",
            flush=True
        )

    iteration += 1


st.leave_sim()

