"""Generate the MuJoCo SITL dataset for the reference study's Koopman attitude model.

reference study protocol retained:
- attitude only
- 40 trajectories, 15 s each
- 0.05 s sampling
- random sinusoidal references, amplitude <= 10 deg and frequency < 0.6 Hz
- 30 training trajectories, 10 validation trajectories
- logged R, body angular velocity, and applied body torque

The reference study does not publish the data-collection controller gains.  The small,
stable PD excitation controller below is therefore an explicit SITL parameter,
not a claimed reference study parameter.  It is intentionally scaled so the generated
trajectories remain in the same small-attitude regime as the reference study's plots.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

SAMPLE_DT = 0.05
TRAJECTORY_DURATION = 15.0
N_TRAJECTORIES = 40
N_TRAIN = 30
N_VALIDATION = 10
AMPLITUDE_LIMIT_DEG = 10.0
FREQUENCY_LIMIT_HZ = 0.6

# Explicit SITL excitation parameters (not published in the reference study).
EXCITATION_KP = np.array([0.06, 0.06, 0.08], dtype=float)
EXCITATION_KD = np.array([0.015, 0.015, 0.020], dtype=float)
EXCITATION_TORQUE_LIMIT = 0.08  # Nm
MAX_STATE_ANGLE_DEG = 20.0      # safety guard against a broken plant/controller


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = map(float, rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = map(float, np.asarray(w).reshape(3))
    return np.array([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]])


def find_free_joint(model: mujoco.MjModel) -> int:
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    ids = np.flatnonzero(np.asarray(model.jnt_type) == free)
    if ids.size == 0:
        raise RuntimeError("MuJoCo XML must contain a free joint for the quadrotor base.")
    return int(ids[0])


def extract_state(model: mujoco.MjModel, data: mujoco.MjData, body_id: int):
    R = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    vel6 = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(
        model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel6, 1
    )
    omega_body = vel6[3:].copy()
    rpy = matrix_to_rpy(R)
    return R, omega_body, rpy


def apply_body_torque(data: mujoco.MjData, body_id: int, R: np.ndarray, tau_body: np.ndarray):
    # xfrc_applied uses the world frame; convert body torque with R.
    data.xfrc_applied[body_id, :] = 0.0
    data.xfrc_applied[body_id, 3:6] = R @ np.asarray(tau_body, dtype=float)


def reset_free_body(model, data, free_joint_id, rpy_rad):
    qpos_adr = int(model.jnt_qposadr[free_joint_id])
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[qpos_adr:qpos_adr + 3] = 0.0
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat_wxyz_from_matrix(rpy_to_matrix(rpy_rad))
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def wrap_error(target, current):
    e = np.asarray(target) - np.asarray(current)
    return (e + np.pi) % (2.0 * np.pi) - np.pi


def generate_reference(rng, n_samples, dt):
    t = np.arange(n_samples, dtype=float) * dt
    ref = np.zeros((n_samples, 3), dtype=float)
    for axis in range(3):
        amplitude_deg = rng.uniform(5.0, 10.0)
        frequency_hz = rng.uniform(0.15, 0.55)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ref[:, axis] = np.deg2rad(amplitude_deg) * np.sin(
            2.0 * np.pi * frequency_hz * t + phase
        )
    return ref


def excitation_torque(reference, current_rpy, omega_body):
    err = wrap_error(reference, current_rpy)
    tau = EXCITATION_KP * err - EXCITATION_KD * omega_body
    return np.clip(tau, -EXCITATION_TORQUE_LIMIT, EXCITATION_TORQUE_LIMIT)


def simulate_trajectory(model, body_id, free_joint_id, reference):
    data = mujoco.MjData(model)
    reset_free_body(model, data, free_joint_id, np.zeros(3))

    n = reference.shape[0]
    R_hist = np.zeros((n, 3, 3), dtype=float)
    omega_hist = np.zeros((n, 3), dtype=float)
    rpy_hist = np.zeros((n, 3), dtype=float)
    tau_hist = np.zeros((n - 1, 3), dtype=float)

    sim_dt = float(model.opt.timestep)
    ratio = SAMPLE_DT / sim_dt
    substeps = int(round(ratio))
    if not np.isclose(ratio, substeps, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            f"MuJoCo timestep {sim_dt} must divide 0.05 exactly; ratio={ratio}."
        )

    for k in range(n):
        R, omega, rpy = extract_state(model, data, body_id)
        R_hist[k] = R
        omega_hist[k] = omega
        rpy_hist[k] = rpy

        max_angle = np.max(np.abs(np.rad2deg(rpy)))
        if max_angle > MAX_STATE_ANGLE_DEG:
            raise RuntimeError(
                f"Trajectory became unstable at t={k * SAMPLE_DT:.3f}s: "
                f"max |angle|={max_angle:.2f} deg. "
                "Check the XML inertia/torque scaling before continuing."
            )

        if k == n - 1:
            break

        tau = excitation_torque(reference[k], rpy, omega)
        tau_hist[k] = tau

        # Recompute R at each substep so the body->world torque transform stays current.
        for _ in range(substeps):
            R_now = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
            apply_body_torque(data, body_id, R_now, tau)
            mujoco.mj_step(model, data)

    return {
        "R": R_hist,
        "omega": omega_hist,
        "rpy": rpy_hist,
        "tau": tau_hist,
        "reference": reference,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("mujoco/quadrotor.xml"))
    p.add_argument("--body-name", type=str, default="base")
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model.resolve())

    n_samples = int(round(TRAJECTORY_DURATION / SAMPLE_DT)) + 1
    rng = np.random.default_rng(args.seed)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    body_id = int(model.body(args.body_name).id)
    free_joint_id = find_free_joint(model)

    train, validation = [], []
    for idx in range(N_TRAJECTORIES):
        reference = generate_reference(rng, n_samples, SAMPLE_DT)
        traj = simulate_trajectory(model, body_id, free_joint_id, reference)
        (train if idx < N_TRAIN else validation).append(traj)
        split = "train" if idx < N_TRAIN else "validation"
        print(f"Generated trajectory {idx + 1:02d}/{N_TRAJECTORIES} ({split})")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def stack(items, key):
        return np.stack([x[key] for x in items], axis=0)

    common = dict(
        sample_dt=np.array(SAMPLE_DT),
        trajectory_duration=np.array(TRAJECTORY_DURATION),
        seed=np.array(args.seed),
    )

    np.savez_compressed(
        args.output_dir / "trajectories_train.npz",
        R=stack(train, "R"), omega=stack(train, "omega"),
        rpy=stack(train, "rpy"), tau=stack(train, "tau"),
        reference=stack(train, "reference"),
        n_trajectories=np.array(N_TRAIN), **common
    )
    np.savez_compressed(
        args.output_dir / "trajectories_val.npz",
        R=stack(validation, "R"), omega=stack(validation, "omega"),
        rpy=stack(validation, "rpy"), tau=stack(validation, "tau"),
        reference=stack(validation, "reference"),
        n_trajectories=np.array(N_VALIDATION), **common
    )

    train_omega = stack(train, "omega")
    train_tau = stack(train, "tau")
    train_rpy = stack(train, "rpy")
    print("\nDataset generation complete.")
    print(f"Training file:   {args.output_dir / 'trajectories_train.npz'}")
    print(f"Validation file: {args.output_dir / 'trajectories_val.npz'}")
    print(f"Training samples: {N_TRAIN} trajectories x {n_samples} states")
    print(f"Max training |omega|: {np.max(np.linalg.norm(train_omega, axis=-1)):.6f} rad/s")
    print(f"Max training |tau|:   {np.max(np.abs(train_tau)):.6f} Nm")
    print(f"Max training |rpy|:   {np.max(np.abs(np.rad2deg(train_rpy))):.6f} deg")


if __name__ == "__main__":
    main()
