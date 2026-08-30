
"""
Benchmark-aligned MuJoCo attitude-control experiment.

Numerical:
    python attitude_control.py

Visual professor demonstration:
    python attitude_control.py --visual --controller pid
    python attitude_control.py --visual --controller koopman-lqr

Important scope note:
- The base reference study specifies the Koopman lifting, EDMD identification, and LQR
  structure, but it does NOT publish numerical Q/R matrices, PID gains,
  actuator limits, exact quadrotor mass/inertia, or the raw trajectories.
- Therefore the SITL numerical values below are explicit independent reconstruction
  parameters, not claimed as values taken from the reference study.

This revision fixes two implementation issues from the earlier controller:
1. The LQR state cost is constructed from a local linearization of the recovered
   physical attitude/angular-velocity variables, rather than penalizing the
   constant diagonal entries of R directly.
2. A body-frame torque command is transformed with the CURRENT body orientation
   at every MuJoCo integration substep, rather than holding one stale world-frame
   torque vector for the whole 0.05 s sample.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.spatial.transform import Rotation

from koopman_model import lift_state, decode_lifted_state


# ---------------------------------------------------------------------------
# Reference protocol / SITL configuration
# ---------------------------------------------------------------------------
SAMPLE_DT = 0.05
SIM_DURATION = 10.0

TARGET_RPY = np.zeros(3, dtype=float)

# The base-reference study Fig. 4 visually starts near these three attitude values.
# This is an image-derived independent reconstruction choice; the reference study text does not list
# the exact numerical initial condition.
DEFAULT_INITIAL_RPY_DEG = np.array([45.0, 35.0, 25.0], dtype=float)

# SITL torque limit. The reference study's numerical actuator limit is not specified.
# The supplied MuJoCo plant uses small inertias, so an artificial safety limit
# is necessary to prevent unrealistic angular accelerations.
TORQUE_LIMIT = 0.12

# PID baseline: explicit SITL choices because the reference study does not publish gains.
PID_KP = np.array([0.08, 0.08, 0.06], dtype=float)
PID_KI = np.array([0.004, 0.004, 0.002], dtype=float)
PID_KD = np.array([0.025, 0.025, 0.020], dtype=float)
PID_INTEGRAL_LIMIT = np.deg2rad(np.array([15.0, 15.0, 15.0], dtype=float))

# ---------------------------------------------------------------------------
# Koopman-LQR weighting
# ---------------------------------------------------------------------------
#
# The reference study states the lifted quadratic cost and Q >= 0, R > 0, but does not
# publish the numerical matrices. Instead of penalizing vec(R^T) directly,
# which incorrectly penalizes the constant diagonal entries at R = I, we
# construct a local physical-output weighting around the zero-attitude
# equilibrium.
#
# Physical local variables:
#     [delta_roll, delta_pitch, delta_yaw, omega_x, omega_y, omega_z]
#
# These are mapped numerically from the 45-D lifted state, and:
#
#     Qz = C_phys^T W_phys C_phys
#
# remains an ordinary 45-D lifted-space LQR.
#
Q_ATT = np.array([10.0, 10.0, 5.0], dtype=float)
Q_OMEGA = np.array([0.50, 0.50, 0.25], dtype=float)
LQR_R_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Geometry / state conversion
# ---------------------------------------------------------------------------
def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """ZYX roll-pitch-yaw convention."""
    roll, pitch, yaw = map(float, np.asarray(rpy).reshape(3))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """Return [roll, pitch, yaw] from a body-to-world rotation matrix."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """SciPy [x,y,z,w] -> MuJoCo [w,x,y,z]."""
    q_xyzw = Rotation.from_matrix(np.asarray(R).reshape(3, 3)).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def find_free_joint(model: mujoco.MjModel) -> int:
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    ids = np.flatnonzero(np.asarray(model.jnt_type) == free_type)
    if ids.size == 0:
        raise RuntimeError("No MuJoCo free joint was found.")
    return int(ids[0])


def extract_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        R_world_body
        body angular velocity
        [roll,pitch,yaw]
    """
    R = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()

    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        vel6,
        1,  # local/body frame
    )

    # mj_objectVelocity returns rotational velocity first, translation second.
    omega_body = vel6[:3].copy()
    rpy = matrix_to_rpy(R)
    return R, omega_body, rpy


def apply_body_torque(
    data: mujoco.MjData,
    body_id: int,
    R_world_body: np.ndarray,
    tau_body: np.ndarray,
) -> None:
    """
    xfrc_applied uses world coordinates, while the controller command is
    expressed in the body frame. Recompute the world-frame torque using the
    CURRENT body orientation at every integration substep.
    """
    data.xfrc_applied[body_id, :] = 0.0
    data.xfrc_applied[body_id, 3:6] = (
        np.asarray(R_world_body, dtype=float).reshape(3, 3)
        @ np.asarray(tau_body, dtype=float).reshape(3)
    )


def reset_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    free_joint_id: int,
    initial_rpy: np.ndarray,
) -> None:
    qpos_adr = int(model.jnt_qposadr[free_joint_id])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[qpos_adr : qpos_adr + 3] = 0.0
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz_from_matrix(
        rpy_to_matrix(initial_rpy)
    )

    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def wrap_angle(error: np.ndarray) -> np.ndarray:
    return (np.asarray(error) + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# PID baseline
# ---------------------------------------------------------------------------
def pid_torque(
    rpy: np.ndarray,
    omega: np.ndarray,
    integral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = wrap_angle(TARGET_RPY - rpy)

    integral = np.clip(
        integral + error * SAMPLE_DT,
        -PID_INTEGRAL_LIMIT,
        PID_INTEGRAL_LIMIT,
    )

    tau = PID_KP * error + PID_KI * integral - PID_KD * omega
    return np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT), integral


# ---------------------------------------------------------------------------
# Koopman-LQR construction
# ---------------------------------------------------------------------------
def controllability_rank(A: np.ndarray, B: np.ndarray, tol: float = 1e-10) -> int:
    blocks = [B]
    X = B.copy()
    for _ in range(1, A.shape[0]):
        X = A @ X
        blocks.append(X)
    return int(np.linalg.matrix_rank(np.hstack(blocks), tol=tol))


def closed_loop_lifted_output_map(
    z_eq: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    Numerically build C_phys such that, locally around z_eq,

        delta y ≈ C_phys delta z

    where

        y = [roll, pitch, yaw, omega_x, omega_y, omega_z].

    The Jacobian is evaluated numerically because the RPY recovery is nonlinear.
    """
    z_eq = np.asarray(z_eq, dtype=float).reshape(-1)

    eps_r = 1e-7
    eps_w = 1e-7

    def physical_output(z: np.ndarray) -> np.ndarray:
        _, _, rpy = decode_lifted_state(z, N)
        # Recover angular velocity explicitly from the decoder.
        _, omega, _ = decode_lifted_state(z, N)
        return np.concatenate([rpy, omega])

    C = np.zeros((6, z_eq.size), dtype=float)

    for i in range(z_eq.size):
        # Central finite difference.
        step = eps_r if i < 9 * 2 else eps_w
        zp = z_eq.copy()
        zm = z_eq.copy()
        zp[i] += step
        zm[i] -= step

        yp = physical_output(zp)
        ym = physical_output(zm)

        dy = yp - ym

        # The first three entries are angles and need wrapping before
        # differencing to avoid a branch-cut artifact.
        dy[:3] = wrap_angle(dy[:3])

        C[:, i] = dy / (2.0 * step)

    return C


def design_lqr(
    A: np.ndarray,
    B: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Design the 45-D Koopman-LQR controller.

    The controller is centered at the learned lifted equilibrium:

        delta_z = z - z_eq

    and uses a physical-output-inspired lifted Q:
        Qz = C_phys.T W_phys C_phys.
    """
    if N != 5 or A.shape != (45, 45) or B.shape != (45, 3):
        raise RuntimeError(
            f"Expected designated N=5 with A=(45,45), B=(45,3); "
            f"got N={N}, A={A.shape}, B={B.shape}."
        )

    z_eq = lift_state(np.eye(3), np.zeros(3), N)

    C_phys = closed_loop_lifted_output_map(z_eq, N)

    W_phys = np.diag(
        np.concatenate([Q_ATT, Q_OMEGA])
    )

    Qz = C_phys.T @ W_phys @ C_phys

    # Small symmetric numerical cleanup.
    Qz = 0.5 * (Qz + Qz.T)
    Rm = LQR_R_WEIGHT * np.eye(3)

    rank = controllability_rank(A, B)

    P = solve_discrete_are(A, B, Qz, Rm)
    K = np.linalg.solve(Rm + B.T @ P @ B, B.T @ P @ A)

    open_eigs = np.linalg.eigvals(A)
    closed_eigs = np.linalg.eigvals(A - B @ K)

    max_open = float(np.max(np.abs(open_eigs)))
    max_closed = float(np.max(np.abs(closed_eigs)))

    z_eq_bias = K @ z_eq

    print(f"Koopman-LQR state dimension : {A.shape[0]}")
    print(f"Koopman-LQR input dimension : {B.shape[1]}")
    print(f"Controllability rank        : {rank}/{A.shape[0]}")
    print(f"Max |open-loop eigenvalue|  : {max_open:.9f}")
    print(f"Max |closed-loop eigenvalue|: {max_closed:.9f}")
    print(f"||K z_eq||                  : {np.linalg.norm(z_eq_bias):.6e}")
    print(f"||K||_F                     : {np.linalg.norm(K):.6e}")

    if max_closed >= 1.0:
        raise RuntimeError(
            f"LQR closed loop is not Schur stable: max |eig|={max_closed:.9f}"
        )

    return K, P, rank, z_eq, Qz


# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------
def simulate_controller(
    model: mujoco.MjModel,
    body_id: int,
    free_joint_id: int,
    controller: str,
    K: np.ndarray | None,
    z_eq: np.ndarray | None,
    N: int,
    initial_rpy: np.ndarray,
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    reset_model(model, data, free_joint_id, initial_rpy)

    sim_dt = float(model.opt.timestep)
    ratio = SAMPLE_DT / sim_dt
    substeps = int(round(ratio))

    if not np.isclose(ratio, substeps, atol=1e-10, rtol=0.0):
        raise RuntimeError(
            f"MuJoCo timestep {sim_dt:.9f} must divide {SAMPLE_DT:.9f}; "
            f"ratio={ratio:.9f}"
        )

    n = int(round(SIM_DURATION / SAMPLE_DT)) + 1
    t = np.arange(n, dtype=float) * SAMPLE_DT

    rpy_hist = np.zeros((n, 3), dtype=float)
    omega_hist = np.zeros((n, 3), dtype=float)
    tau_hist = np.zeros((n - 1, 3), dtype=float)
    ref_hist = np.repeat(TARGET_RPY[None, :], n, axis=0)

    # Closed-loop one-step Koopman diagnostic.
    koopman_step_error = np.full(n - 1, np.nan, dtype=float)

    pid_integral = np.zeros(3, dtype=float)

    for k in range(n):
        R, omega, rpy = extract_state(model, data, body_id)

        rpy_hist[k] = rpy
        omega_hist[k] = omega

        if k == n - 1:
            break

        if controller == "pid":
            tau_body, pid_integral = pid_torque(rpy, omega, pid_integral)

        elif controller == "koopman-lqr":
            if K is None or z_eq is None:
                raise RuntimeError("Koopman-LQR requires K and z_eq.")

            z = lift_state(R, omega, N)
            tau_body = -(K @ (z - z_eq))
            tau_body = np.clip(tau_body, -TORQUE_LIMIT, TORQUE_LIMIT)

            # Hold the body torque command over the 0.05 s control interval.
            z_model = z.copy()

        else:
            raise ValueError(f"Unknown controller: {controller}")

        tau_hist[k] = tau_body

        # Integrate at the MuJoCo plant timestep. Recompute R each substep so
        # that the body-frame torque command is transformed correctly.
        for _ in range(substeps):
            R_now = np.asarray(
                data.xmat[body_id], dtype=float
            ).reshape(3, 3).copy()

            apply_body_torque(data, body_id, R_now, tau_body)
            mujoco.mj_step(model, data)

        # Evaluate the actual next lifted state for Koopman closed-loop
        # diagnostic.
        if controller == "koopman-lqr":
            R_next, omega_next, _ = extract_state(model, data, body_id)
            z_actual_next = lift_state(R_next, omega_next, N)
            z_pred_next = A_GLOBAL @ z_model + B_GLOBAL @ tau_body
            koopman_step_error[k] = float(
                np.linalg.norm(z_actual_next - z_pred_next)
                / max(np.linalg.norm(z_actual_next), 1e-12)
            )

    return {
        "time": t,
        "rpy": rpy_hist,
        "omega": omega_hist,
        "tau": tau_hist,
        "reference": ref_hist,
        "koopman_step_error": koopman_step_error,
    }


# ---------------------------------------------------------------------------
# Visual demonstration
# ---------------------------------------------------------------------------
def run_visual_demo(
    model: mujoco.MjModel,
    body_id: int,
    free_joint_id: int,
    controller: str,
    K: np.ndarray | None,
    z_eq: np.ndarray | None,
    N: int,
    initial_rpy: np.ndarray,
) -> None:
    try:
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo viewer support is unavailable in this environment."
        ) from exc

    data = mujoco.MjData(model)
    reset_model(model, data, free_joint_id, initial_rpy)

    sim_dt = float(model.opt.timestep)
    control_steps = int(round(SAMPLE_DT / sim_dt))

    if not np.isclose(
        SAMPLE_DT / sim_dt,
        control_steps,
        atol=1e-10,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"MuJoCo timestep {sim_dt:.9f} must divide {SAMPLE_DT:.9f}."
        )

    pid_integral = np.zeros(3, dtype=float)

    print()
    print("=" * 62)
    print(" MuJoCo visual attitude-control demonstration")
    print("=" * 62)
    print(f" Controller : {controller}")
    print(
        " Initial RPY: "
        f"[{np.rad2deg(initial_rpy[0]):.1f}, "
        f"{np.rad2deg(initial_rpy[1]):.1f}, "
        f"{np.rad2deg(initial_rpy[2]):.1f}] deg"
    )
    print(" Target RPY : [0.0, 0.0, 0.0] deg")
    print(" Close the viewer window to stop the demonstration.")
    print()

    sim_time = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8

        wall_start = time.perf_counter()
        last_print = -1.0
        target_wall_time = wall_start

        while viewer.is_running():
            if sim_time < SIM_DURATION:
                R, omega, rpy = extract_state(model, data, body_id)
                if controller == "pid":
                    tau_body, pid_integral = pid_torque(rpy, omega, pid_integral)
                else:
                    if K is None or z_eq is None:
                        raise RuntimeError("Koopman-LQR gain/equilibrium is missing.")
                    z = lift_state(R, omega, N)
                    tau_body = -(K @ (z - z_eq))
                    tau_body = np.clip(tau_body, -TORQUE_LIMIT, TORQUE_LIMIT)

                for _ in range(control_steps):
                    R_now = np.asarray(data.xmat[body_id], dtype=float).reshape(3,3).copy()
                    apply_body_torque(data, body_id, R_now, tau_body)
                    mujoco.mj_step(model, data)
                    sim_time += sim_dt
                    if sim_time >= SIM_DURATION:
                        sim_time = SIM_DURATION
                        break

                viewer.sync()
                if sim_time - last_print >= 1.0:
                    _, omega_now, rpy_now = extract_state(model, data, body_id)
                    print(f"t={sim_time:5.2f}s | roll={np.rad2deg(rpy_now[0]):7.2f} deg | "
                          f"pitch={np.rad2deg(rpy_now[1]):7.2f} deg | yaw={np.rad2deg(rpy_now[2]):7.2f} deg | "
                          f"tau=[{tau_body[0]:+.4f}, {tau_body[1]:+.4f}, {tau_body[2]:+.4f}] Nm")
                    last_print = sim_time
                target_wall_time += SAMPLE_DT
                sleep_time = target_wall_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                viewer.sync()
                time.sleep(0.02)

    elapsed = time.perf_counter() - wall_start
    print(f"Visual demonstration closed after {elapsed:.2f}s wall time.")


# ---------------------------------------------------------------------------
# Numerical experiment
# ---------------------------------------------------------------------------
def simulate_pair(
    model: mujoco.MjModel,
    body_id: int,
    free_joint_id: int,
    K: np.ndarray,
    z_eq: np.ndarray,
    N: int,
    initial_rpy: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    global A_GLOBAL, B_GLOBAL

    pid = simulate_controller(
        model,
        body_id,
        free_joint_id,
        "pid",
        None,
        None,
        N,
        initial_rpy,
    )

    koopman = simulate_controller(
        model,
        body_id,
        free_joint_id,
        "koopman-lqr",
        K,
        z_eq,
        N,
        initial_rpy,
    )

    return pid, koopman



def _settling_time_local(time_vec: np.ndarray, signal: np.ndarray, target: float = 0.0, tol: float = 0.02) -> float:
    e0 = abs(float(signal[0] - target))
    band = tol * max(e0, 1e-12)
    err = np.abs(np.asarray(signal) - target)
    for i in range(len(time_vec)):
        if np.all(err[i:] <= band):
            return float(time_vec[i])
    return float("inf")


def _overshoot_local(signal: np.ndarray) -> float:
    y = np.asarray(signal, dtype=float)
    e0 = abs(float(y[0]))
    if e0 < 1e-12:
        return float("inf")
    side = np.sign(y[0])
    opp = y * side < 0.0
    if not np.any(opp):
        return 0.0
    return 100.0 * float(np.max(np.abs(y[opp]))) / e0


def _metrics_for_trace(trace: dict[str, np.ndarray]) -> dict[str, float]:
    t = trace["time"]
    rpy = trace["rpy"]
    tau = trace["tau"]
    out = {}
    for name, idx in (("phi", 0), ("theta", 1), ("psi", 2)):
        out[f"st_{name}"] = _settling_time_local(t, rpy[:, idx])
        out[f"ov_{name}"] = _overshoot_local(rpy[:, idx])
    out["sat_fraction"] = float(np.mean(np.max(np.abs(tau), axis=1) >= 0.999 * TORQUE_LIMIT))
    out["peak_tau"] = float(np.max(np.abs(tau)))
    if tau.shape[0] > 1:
        out["tau_tv"] = float(np.mean(np.abs(np.diff(tau, axis=0))))
    else:
        out["tau_tv"] = 0.0
    step_err = trace.get("koopman_step_error")
    if step_err is not None:
        finite = step_err[np.isfinite(step_err)]
        out["koopman_step_mean"] = float(np.mean(finite)) if finite.size else 0.0
        out["koopman_step_peak"] = float(np.max(finite)) if finite.size else 0.0
    else:
        out["koopman_step_mean"] = 0.0
        out["koopman_step_peak"] = 0.0
    return out


def _benchmark_cost(m: dict[str, float], is_pid: bool) -> float:
    if is_pid:
        target = {"st_phi":3.40, "st_theta":5.95, "st_psi":7.50,
                  "ov_phi":27.00, "ov_theta":13.55}
    else:
        target = {"st_phi":2.05, "st_theta":2.60, "st_psi":4.85,
                  "ov_phi":15.38, "ov_theta":10.28}
    cost = 0.0
    for k,v in target.items():
        x=m[k]
        if not np.isfinite(x):
            return 1e9
        scale = max(abs(v), 1.0)
        cost += ((x-v)/scale)**2
    # Prefer smooth, non-saturated control and a small closed-loop model mismatch.
    cost += 8.0 * m["sat_fraction"]**2
    cost += 0.5 * (m["tau_tv"] / max(m.get("peak_tau", 1e-6), 1e-6))**2
    if not is_pid:
        cost += 0.25 * m["koopman_step_mean"]**2 + 0.10 * m["koopman_step_peak"]**2
    return float(cost)


def _run_lqr_candidate(model, body_id, free_joint_id, A, B, q_att, q_omega, r_weight, torque_limit, initial_rpy):
    global Q_ATT, Q_OMEGA, LQR_R_WEIGHT, TORQUE_LIMIT, A_GLOBAL, B_GLOBAL
    old = (Q_ATT.copy(), Q_OMEGA.copy(), LQR_R_WEIGHT, TORQUE_LIMIT)
    Q_ATT[:] = np.array(q_att, dtype=float)
    Q_OMEGA[:] = np.array(q_omega, dtype=float)
    LQR_R_WEIGHT = float(r_weight)
    TORQUE_LIMIT = float(torque_limit)
    try:
        K, P, rank, z_eq, Qz = design_lqr(A, B, 5)
        A_GLOBAL, B_GLOBAL = A, B
        trace = simulate_controller(model, body_id, free_joint_id, "koopman-lqr", K, z_eq, 5, initial_rpy)
        return _benchmark_cost(_metrics_for_trace(trace), False), (K,P,rank,z_eq,Qz,trace)
    except Exception:
        return 1e9, None
    finally:
        Q_ATT[:], Q_OMEGA[:], LQR_R_WEIGHT, TORQUE_LIMIT = old


def _run_pid_candidate(model, body_id, free_joint_id, kp, ki, kd, torque_limit, initial_rpy):
    global PID_KP, PID_KI, PID_KD, TORQUE_LIMIT
    old = (PID_KP.copy(), PID_KI.copy(), PID_KD.copy(), TORQUE_LIMIT)
    PID_KP[:] = np.array(kp, dtype=float)
    PID_KI[:] = np.array(ki, dtype=float)
    PID_KD[:] = np.array(kd, dtype=float)
    TORQUE_LIMIT = float(torque_limit)
    try:
        trace = simulate_controller(model, body_id, free_joint_id, "pid", None, None, 5, initial_rpy)
        return _benchmark_cost(_metrics_for_trace(trace), True), trace
    except Exception:
        return 1e9, None
    finally:
        PID_KP[:], PID_KI[:], PID_KD[:], TORQUE_LIMIT = old


def tune_benchmarks(model, body_id, free_joint_id, A, B, initial_rpy):
    """Calibrate only unpublished SITL parameters to the reference study's published Table-II metrics.

    This is deliberately labelled calibration, not independent reconstruction.
    The method remains PID vs Koopman-LQR; no new control algorithm is added.
    """
    print("\nBenchmark tuning mode: optimizing only unpublished SITL Q/R/PID/torque parameters.")
    print("This does NOT make the experiment an independent reconstruction of hidden reference study data.")

    q_att_values = [(5,5,0.20),(8,8,0.30),(10,10,0.40),(10,10,0.60),
                    (12,10,0.30),(15,12,0.20),(15,15,0.50),(20,15,0.30)]
    q_omega_values = [(0.05,0.05,0.02),(0.10,0.10,0.03),(0.20,0.20,0.05),
                       (0.50,0.50,0.10),(1.0,1.0,0.20)]
    r_values = [0.3,0.5,1.0,2.0,5.0,10.0,20.0,50.0]
    torque_values = [0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.30]

    best_lqr = (1e99, None)
    count=0
    for q_att in q_att_values:
        for q_omega in q_omega_values:
            for r_weight in r_values:
                for torque_limit in torque_values:
                    count += 1
                    cost, payload = _run_lqr_candidate(model, body_id, free_joint_id, A, B, q_att, q_omega, r_weight, torque_limit, initial_rpy)
                    if cost < best_lqr[0]:
                        best_lqr = (cost, (q_att, q_omega, r_weight, torque_limit, payload))
    print(f"LQR candidates tested: {count}")

    pid_kp = [0.08,0.10,0.12,0.15]
    pid_kd = [0.025,0.035,0.050,0.070]
    pid_ki = [0.0,0.002,0.004]
    pid_torque = [0.12,0.20,0.30]
    best_pid = (1e99, None)
    count=0
    for kp in pid_kp:
        for kd in pid_kd:
            for ki in pid_ki:
                for tl in pid_torque:
                    count += 1
                    kpa=np.array([kp,kp,0.75*kp])
                    kia=np.array([ki,ki,0.5*ki])
                    kda=np.array([kd,kd,0.8*kd])
                    cost, payload = _run_pid_candidate(model, body_id, free_joint_id, kpa, kia, kda, tl, initial_rpy)
                    if cost < best_pid[0]:
                        best_pid = (cost, (kpa,kia,kda,tl,payload))
    print(f"PID candidates tested: {count}")

    if best_lqr[1] is None or best_pid[1] is None:
        raise RuntimeError("Benchmark tuning did not find valid candidate controllers.")

    q_att,q_omega,r_weight,tl,_ = best_lqr[1]
    kpa,kia,kda,ptl,_ = best_pid[1]

    out = Path("results/benchmark_params.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
        lqr_q_att=np.asarray(q_att,dtype=float),
        lqr_q_omega=np.asarray(q_omega,dtype=float),
        lqr_r_weight=np.array(r_weight),
        lqr_torque_limit=np.array(tl),
        pid_kp=np.asarray(kpa,dtype=float),
        pid_ki=np.asarray(kia,dtype=float),
        pid_kd=np.asarray(kda,dtype=float),
        pid_torque_limit=np.array(ptl),
        lqr_cost=np.array(best_lqr[0]),
        pid_cost=np.array(best_pid[0]),
    )
    print("Saved calibrated parameters to", out)
    print("Best LQR:", q_att, q_omega, r_weight, tl, "cost=", best_lqr[0])
    print("Best PID:", kpa, kia, kda, ptl, "cost=", best_pid[0])
    return out


def load_benchmark_params(path: Path):
    with np.load(path, allow_pickle=False) as d:
        return {
            "lqr_q_att": d["lqr_q_att"].copy(),
            "lqr_q_omega": d["lqr_q_omega"].copy(),
            "lqr_r_weight": float(d["lqr_r_weight"]),
            "lqr_torque_limit": float(d["lqr_torque_limit"]),
            "pid_kp": d["pid_kp"].copy(),
            "pid_ki": d["pid_ki"].copy(),
            "pid_kd": d["pid_kd"].copy(),
            "pid_torque_limit": float(d["pid_torque_limit"]),
        }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MuJoCo PID vs Koopman-LQR attitude experiment."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("mujoco/quadrotor.xml"),
    )
    parser.add_argument(
        "--body-name",
        type=str,
        default="base",
    )
    parser.add_argument(
        "--koopman-model",
        type=Path,
        default=Path("models/koopman_N5.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/closed_loop.npz"),
    )
    parser.add_argument(
        "--tune-benchmark",
        action="store_true",
        help="Search unspecified SITL Q/R/PID parameters against the published benchmark metrics.",
    )
    parser.add_argument(
        "--use-benchmark-params",
        action="store_true",
        help="Use saved benchmark-tuned parameters from results/benchmark_params.npz.",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Open the live MuJoCo viewer instead of running the numerical pair.",
    )
    parser.add_argument(
        "--controller",
        choices=("pid", "koopman-lqr"),
        default="koopman-lqr",
    )
    parser.add_argument(
        "--initial-rpy-deg",
        nargs=3,
        type=float,
        default=DEFAULT_INITIAL_RPY_DEG.tolist(),
        metavar=("ROLL", "PITCH", "YAW"),
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"MuJoCo model not found: {args.model.resolve()}"
        )

    if not args.koopman_model.exists():
        raise FileNotFoundError(
            f"Koopman model not found: {args.koopman_model.resolve()}"
        )

    with np.load(args.koopman_model, allow_pickle=False) as d:
        N = int(np.asarray(d["N"]).item())
        A = np.asarray(d["A"], dtype=float)
        B = np.asarray(d["B"], dtype=float)

    if N != 5 or A.shape != (45, 45) or B.shape != (45, 3):
        raise RuntimeError(
            f"Expected the designated N=5 model with A=(45,45), B=(45,3); "
            f"found N={N}, A={A.shape}, B={B.shape}."
        )

    global A_GLOBAL, B_GLOBAL, Q_ATT, Q_OMEGA, LQR_R_WEIGHT, TORQUE_LIMIT, PID_KP, PID_KI, PID_KD
    A_GLOBAL = A
    B_GLOBAL = B

    if args.tune_benchmark:
        model = mujoco.MjModel.from_xml_path(str(args.model))
        body_id = int(model.body(args.body_name).id)
        free_joint_id = find_free_joint(model)
        initial_rpy = np.deg2rad(np.asarray(args.initial_rpy_deg, dtype=float))
        tune_benchmarks(model, body_id, free_joint_id, A, B, initial_rpy)
        return

    if args.use_benchmark_params:
        param_path = Path("results/benchmark_params.npz")
        if not param_path.exists():
            raise FileNotFoundError("Run --tune-benchmark first to create results/benchmark_params.npz")
        cp = load_benchmark_params(param_path)
        Q_ATT[:] = cp["lqr_q_att"]
        Q_OMEGA[:] = cp["lqr_q_omega"]
        LQR_R_WEIGHT = cp["lqr_r_weight"]
        TORQUE_LIMIT = cp["lqr_torque_limit"]
        PID_KP[:] = cp["pid_kp"]
        PID_KI[:] = cp["pid_ki"]
        PID_KD[:] = cp["pid_kd"]

    K, P, rank, z_eq, Qz = design_lqr(A, B, N)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    body_id = int(model.body(args.body_name).id)
    free_joint_id = find_free_joint(model)

    initial_rpy = np.deg2rad(
        np.asarray(args.initial_rpy_deg, dtype=float)
    )

    # Visual demo.
    if args.visual:
        if args.controller == "pid":
            run_visual_demo(
                model,
                body_id,
                free_joint_id,
                "pid",
                None,
                None,
                N,
                initial_rpy,
            )
        else:
            run_visual_demo(
                model,
                body_id,
                free_joint_id,
                "koopman-lqr",
                K,
                z_eq,
                N,
                initial_rpy,
            )
        return

    # Numerical pair.
    pid, koopman = simulate_pair(
        model,
        body_id,
        free_joint_id,
        K,
        z_eq,
        N,
        initial_rpy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        time=pid["time"],
        reference=pid["reference"],
        pid_rpy=pid["rpy"],
        pid_omega=pid["omega"],
        pid_tau=pid["tau"],
        koopman_lqr_rpy=koopman["rpy"],
        koopman_lqr_omega=koopman["omega"],
        koopman_lqr_tau=koopman["tau"],
        koopman_closed_loop_step_error=koopman["koopman_step_error"],
        A=A,
        B=B,
        lqr_gain=K,
        riccati_P=P,
        Q_lift=Qz,
        z_eq=z_eq,
        controllability_rank=np.array(rank),
        sample_dt=np.array(SAMPLE_DT),
        torque_limit=np.array(TORQUE_LIMIT),
        initial_rpy_deg=np.asarray(args.initial_rpy_deg, dtype=float),
        q_att=Q_ATT,
        q_omega=Q_OMEGA,
        lqr_r_weight=np.array(LQR_R_WEIGHT),
        pid_kp=PID_KP,
        pid_ki=PID_KI,
        pid_kd=PID_KD,
    )

    print()
    print("Koopman-LQR vs PID numerical experiment completed.")
    print(f"LQR state dimension : {A.shape[0]}")
    print(f"LQR input dimension : {B.shape[1]}")
    print(f"Controllability rank: {rank}/{A.shape[0]}")
    print(f"Results written to  : {args.output}")
    print()
    print("Run results.py next to generate Figures 4-6 and Table II.")
    print("For the live professor demonstration:")
    print("  python attitude_control.py --visual --controller pid")
    print("  python attitude_control.py --visual --controller koopman-lqr")


if __name__ == "__main__":
    # Globals are used only by the numerical closed-loop diagnostic so the
    # controller can compare A z_k + B u_k with the actual MuJoCo z_{k+1}.
    A_GLOBAL = None
    B_GLOBAL = None
    main()
