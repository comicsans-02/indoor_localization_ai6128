"""Evaluate a simple IMU/Kalman waypoint predictor.

For each trace, the filter starts at the first ground-truth waypoint.  At every
later waypoint timestamp it:

1. predicts forward using all intervening accelerometer samples;
2. records the predicted position and its error;
3. corrects with the now-observed waypoint for the next segment.

The error is therefore measured before the current waypoint is supplied to the
filter.  This is an anchored, one-segment-ahead baseline, not fully autonomous
indoor navigation.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation

from dataset.io_f import read_data_file


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "dataset" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "sensor_fusion_viz"
GRAVITY = 9.80665


@dataclass
class TraceResult:
    actual: np.ndarray
    predicted: np.ndarray
    errors: np.ndarray


def rotation_vectors_at_times(ahrs: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Interpolate Android rotation vectors and return normalized quaternions."""
    vector_part = np.column_stack(
        [np.interp(timestamps, ahrs[:, 0], ahrs[:, column]) for column in range(1, 4)]
    )
    scalar_part = np.sqrt(
        np.clip(1.0 - np.sum(vector_part**2, axis=1), a_min=0.0, a_max=None)
    )
    quaternions = np.column_stack((vector_part, scalar_part))
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / np.where(norms == 0, 1.0, norms)


def world_horizontal_acceleration(sensor_data) -> tuple[np.ndarray, np.ndarray]:
    """Rotate device acceleration into the world frame and remove static bias."""
    accelerometer = sensor_data.acce.astype(float)
    ahrs = sensor_data.ahrs.astype(float)
    timestamps = accelerometer[:, 0]

    quaternions = rotation_vectors_at_times(ahrs, timestamps)
    world_acceleration = Rotation.from_quat(quaternions).apply(accelerometer[:, 1:4])
    world_acceleration[:, 2] -= GRAVITY

    horizontal = world_acceleration[:, :2]
    horizontal -= np.median(horizontal, axis=0)

    # Phone accelerometers occasionally contain sharp outliers.  This generous
    # bound limits catastrophic integration without removing normal walking motion.
    horizontal = np.clip(horizontal, -8.0, 8.0)
    return timestamps, horizontal


def process_covariance(dt: float, acceleration_sigma: float = 1.5) -> np.ndarray:
    """White-acceleration process covariance for [x, y, vx, vy]."""
    control = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    covariance = control @ (np.eye(2) * acceleration_sigma**2) @ control.T
    return covariance + np.eye(4) * 1e-9


def predict_without_observation(
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PyKalman's linear prediction equations without wrapper overhead."""
    velocity_decay = np.exp(-0.15 * dt)
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, velocity_decay, 0.0],
            [0.0, 0.0, 0.0, velocity_decay],
        ]
    )
    offset = np.array(
        [
            0.5 * acceleration[0] * dt**2,
            0.5 * acceleration[1] * dt**2,
            acceleration[0] * dt,
            acceleration[1] * dt,
        ]
    )
    predicted_mean = transition @ state_mean + offset
    predicted_covariance = (
        transition @ state_covariance @ transition.T + process_covariance(dt)
    )
    return predicted_mean, predicted_covariance


def correct_with_waypoint(
    kalman_filter: KalmanFilter,
    state_mean: np.ndarray,
    state_covariance: np.ndarray,
    waypoint_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct position after scoring, ready for the following segment."""
    return kalman_filter.filter_update(
        state_mean,
        state_covariance,
        observation=waypoint_xy,
        transition_matrix=np.eye(4),
        transition_offset=np.zeros(4),
        transition_covariance=np.eye(4) * 1e-9,
    )


def estimate_trace(trace_path: Path) -> TraceResult | None:
    """Estimate one trace at its waypoint timestamps."""
    sensor_data = read_data_file(trace_path)
    waypoints = sensor_data.waypoint.astype(float)
    if len(waypoints) < 2 or len(sensor_data.acce) < 2 or len(sensor_data.ahrs) < 2:
        return None

    acceleration_times, horizontal_acceleration = world_horizontal_acceleration(
        sensor_data
    )
    observation_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    kalman_filter = KalmanFilter(
        n_dim_state=4,
        n_dim_obs=2,
        observation_matrices=observation_matrix,
        observation_covariance=np.eye(2) * 0.5**2,
    )

    actual = waypoints[:, 1:3]
    predicted = np.empty_like(actual)
    predicted[0] = actual[0]

    state_mean = np.array([actual[0, 0], actual[0, 1], 0.0, 0.0])
    state_covariance = np.diag([0.1, 0.1, 1.0, 1.0])
    current_time = waypoints[0, 0]
    acceleration_index = int(
        np.searchsorted(acceleration_times, current_time, side="right")
    )

    for waypoint_index in range(1, len(waypoints)):
        target_time = waypoints[waypoint_index, 0]
        while (
            acceleration_index < len(acceleration_times)
            and acceleration_times[acceleration_index] <= target_time
        ):
            sample_time = acceleration_times[acceleration_index]
            dt = (sample_time - current_time) / 1000.0
            if dt > 0:
                state_mean, state_covariance = predict_without_observation(
                    state_mean,
                    state_covariance,
                    horizontal_acceleration[acceleration_index],
                    dt,
                )
                current_time = sample_time
            acceleration_index += 1

        remaining_dt = (target_time - current_time) / 1000.0
        if remaining_dt > 0:
            state_mean, state_covariance = predict_without_observation(
                state_mean,
                state_covariance,
                np.zeros(2),
                remaining_dt,
            )
            current_time = target_time

        predicted[waypoint_index] = state_mean[:2]

        # Score first; only then use this waypoint to anchor the next interval.
        state_mean, state_covariance = correct_with_waypoint(
            kalman_filter,
            state_mean,
            state_covariance,
            actual[waypoint_index],
        )

    errors = np.linalg.norm(predicted[1:] - actual[1:], axis=1)
    return TraceResult(actual=actual, predicted=predicted, errors=errors)


def load_floor_dimensions(floor_dir: Path) -> tuple[float, float]:
    with (floor_dir / "floor_info.json").open(encoding="utf-8") as file:
        map_info = json.load(file)["map_info"]
    return float(map_info["width"]), float(map_info["height"])


def evaluate_floor(floor_dir: Path, output_dir: Path) -> dict[str, float | int | str]:
    """Evaluate all usable traces on one floor and save an aggregate plot."""
    trace_dir = floor_dir / "path_data_files"
    results = []
    for trace_path in sorted(trace_dir.glob("*.txt")):
        result = estimate_trace(trace_path)
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError(f"No usable traces found in {trace_dir}")

    errors = np.concatenate([result.errors for result in results])
    floor_width, floor_height = load_floor_dimensions(floor_dir)
    site = floor_dir.parent.name
    floor = floor_dir.name

    scale = 10.0 / max(floor_width, floor_height)
    figure, axes = plt.subplots(
        figsize=(max(5.0, floor_width * scale), max(5.0, floor_height * scale))
    )
    for result in results:
        axes.plot(
            result.actual[:, 0],
            result.actual[:, 1],
            color="#1b9e77",
            linewidth=1.0,
            alpha=0.35,
        )
        axes.plot(
            result.predicted[:, 0],
            result.predicted[:, 1],
            color="#d95f02",
            linewidth=1.0,
            linestyle="--",
            alpha=0.35,
        )

    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    axes.set_xlim(0, floor_width)
    axes.set_ylim(0, floor_height)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("X position (metres)")
    axes.set_ylabel("Y position (metres)")
    axes.set_title(
        f"{site}/{floor}: actual vs predicted waypoints\n"
        f"Mean error {mean_error:.2f} m | Median {median_error:.2f} m"
    )
    axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes.legend(
        handles=[
            Line2D([0], [0], color="#1b9e77", label="Actual"),
            Line2D([0], [0], color="#d95f02", linestyle="--", label="Predicted"),
        ]
    )

    save_path = output_dir / site / f"{floor}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    return {
        "site": site,
        "floor": floor,
        "traces": len(results),
        "predicted_waypoints": len(errors),
        "mean_error_m": mean_error,
        "median_error_m": median_error,
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "plot": str(save_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a simple IMU/Kalman predictor at waypoint timestamps."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--site", default="site1")
    parser.add_argument("--floors", nargs="+", default=["B1", "F2", "F4"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    summaries = []
    for floor in args.floors:
        floor_dir = data_dir / args.site / floor
        if not floor_dir.is_dir():
            raise FileNotFoundError(f"Floor directory does not exist: {floor_dir}")
        summary = evaluate_floor(floor_dir, output_dir)
        summaries.append(summary)
        print(
            f"{summary['site']}/{summary['floor']}: "
            f"mean={summary['mean_error_m']:.2f} m, "
            f"median={summary['median_error_m']:.2f} m, "
            f"predictions={summary['predicted_waypoints']}"
        )

    summary_path = output_dir / args.site / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
