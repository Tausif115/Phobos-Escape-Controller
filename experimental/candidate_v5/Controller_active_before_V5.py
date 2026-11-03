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

    goalSpeed_m_s = 12.0
    goalVelocity = goalSpeed_m_s * goalDirection

    # Velocity feedback in the Phobos-fixed frame.
    velocity_error = goalVelocity - vel

    accel_gain = 1.0
    accel_phobos = accel_gain * velocity_error

    # Official linear acceleration limit: 0.5 m/s^2.
    max_linear_accel = 0.5
    accel_norm = float(np.linalg.norm(accel_phobos))

    if accel_norm > max_linear_accel:
        accel_phobos *= max_linear_accel / accel_norm

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

