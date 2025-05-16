import io
import json
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from PIL import Image as PilImage


def calculate_stats(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0  # avg, min, max

    _sum = sum(values)
    _len = len(values)
    _min = min(values)
    _max = max(values)

    avg = _sum / _len
    return avg, _min, _max


def generate_preview_image(
    sequence: list[list[float]], width_px: int = 400, height_px: int = 400
) -> PilImage.Image:
    if not sequence:
        return PilImage.new("RGB", (width_px, height_px), color="white")

    # Pre-calculate min/max for normalization from the entire sequence
    all_forces = [p[3] for p in sequence if len(p) > 3]
    all_altitudes = [p[4] for p in sequence if len(p) > 4]

    min_force_seq = min(all_forces) if all_forces else 0.0
    max_force_seq = max(all_forces) if all_forces else 1.0
    if max_force_seq < min_force_seq:
        max_force_seq = min_force_seq

    min_alt_seq = min(all_altitudes) if all_altitudes else 0.0
    max_alt_seq = max(all_altitudes) if all_altitudes else 1.0
    if max_alt_seq < min_alt_seq:
        max_alt_seq = min_alt_seq

    range_force = max_force_seq - min_force_seq
    range_alt = max_alt_seq - min_alt_seq
    if range_force < 1e-6:
        range_force = 1.0
    if range_alt < 1e-6:
        range_alt = 1.0

    fig, ax = plt.subplots(figsize=(width_px / 100.0, height_px / 100.0), dpi=100)

    last_time = -1.0
    time_threshold = 1 / 120
    segment_points = []

    for idx, point_data in enumerate(sequence):
        if len(point_data) < 7:
            continue
        current_time, norm_x, norm_y, force, altitude, azimuth_sin, azimuth_cos = (
            point_data
        )
        if last_time >= 0 and (current_time - last_time) > time_threshold:
            segment_points = []
        segment_points.append(
            [norm_x, norm_y, force, altitude, azimuth_sin, azimuth_cos]
        )
        if len(segment_points) >= 2:
            p0 = segment_points[-2]
            p1 = segment_points[-1]
            # Interpolate properties for the segment
            for i, (start, end) in enumerate(zip(p0, p1)):
                pass  # just for clarity
            # Use the average of the two points for properties
            avg_force = (p0[2] + p1[2]) / 2
            avg_altitude = (p0[3] + p1[3]) / 2
            avg_az_sin = (p0[4] + p1[4]) / 2
            avg_az_cos = (p0[5] + p1[5]) / 2
            avg_az_rad = math.atan2(avg_az_sin, avg_az_cos)
            norm_f = (
                (avg_force - min_force_seq) / range_force if range_force > 1e-6 else 0.5
            )
            l_width = 0.5 + norm_f * 2.0
            norm_a = (
                (avg_altitude - min_alt_seq) / range_alt if range_alt > 1e-6 else 0.5
            )
            val = 0.4 + norm_a * 0.6
            hue_val = (avg_az_rad + math.pi) / (2 * math.pi)
            sat = 0.9
            seg_color = mcolors.hsv_to_rgb((hue_val, sat, val)).tolist()
            seg_color = [max(0, min(1, c)) for c in seg_color]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=seg_color,
                linewidth=l_width,
                solid_capstyle="round",
            )
        last_time = current_time

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    buf.seek(0)
    img = PilImage.open(buf).convert("RGB")
    return img


def load_local_dataset() -> DatasetDict:
    """
    Load local datasets from the 'data' directory.
    Handles duplicate items by keeping only the first encountered instance based on 'createdAt'.
    Normalizes X, Y coordinates and offsets timestamps to start from 0.
    Calculates width, height, and other statistics for each signature.
    Generates a preview image for each signature.
    Decomposes azimuth into sin and cos components.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    processed_records = []
    seen_created_at = set()

    for file_path in data_dir.glob("*.jsonl"):
        signer = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw_data = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Skipping malformed JSON line in {file_path}: {line.strip()}"
                    )
                    continue

                created_at = raw_data.get("createdAt")
                if not created_at:
                    print(
                        f"Skipping record without 'createdAt' in {file_path}: {line.strip()}"
                    )
                    continue

                if created_at in seen_created_at:
                    continue
                seen_created_at.add(created_at)

                samples = raw_data.get("samples")
                if not samples or not isinstance(samples, list) or len(samples) == 0:
                    print(
                        f"Skipping record with no/invalid samples for createdAt {created_at} in {file_path}"
                    )
                    continue

                (
                    _timestamps,
                    _xs,
                    _ys,
                    _forces,
                    _altitudes,
                    _azimuths_sin,
                    _azimuths_cos,
                ) = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                valid_record_samples = True
                for s_data in samples:
                    if not isinstance(s_data, dict):
                        valid_record_samples = False
                        break

                    loc = s_data.get("location")
                    ts = s_data.get("timestamp")
                    force = s_data.get("force")
                    azimuth = s_data.get("azimuth", 0.0)  # Get raw azimuth

                    if not (
                        isinstance(loc, list)
                        and len(loc) >= 2
                        and isinstance(ts, (int, float))
                        and isinstance(force, (int, float))
                        and isinstance(
                            azimuth, (int, float)
                        )  # Ensure azimuth is a number
                    ):
                        valid_record_samples = False
                        break

                    _timestamps.append(ts)
                    _xs.append(loc[0])
                    _ys.append(loc[1])
                    _forces.append(force)
                    _altitudes.append(s_data.get("altitude", 0.0))
                    _azimuths_sin.append(
                        math.sin(azimuth)
                    )  # Calculate and store sin(azimuth)
                    _azimuths_cos.append(
                        math.cos(azimuth)
                    )  # Calculate and store cos(azimuth)

                if not valid_record_samples or not _timestamps:
                    print(
                        f"Skipping record with malformed or empty sample data for createdAt {created_at} in {file_path}"
                    )
                    continue

                timestamps, xs, ys, forces, altitudes, azimuths_sin, azimuths_cos = (
                    _timestamps,
                    _xs,
                    _ys,
                    _forces,
                    _altitudes,
                    _azimuths_sin,
                    _azimuths_cos,
                )

                first_timestamp = timestamps[0]

                min_x_val, max_x_val = min(xs), max(xs)
                min_y_val, max_y_val = min(ys), max(ys)

                width = max_x_val - min_x_val
                height = max_y_val - min_y_val

                sequence = []
                for i in range(len(timestamps)):
                    t_offset = timestamps[i] - first_timestamp

                    current_x = xs[i]
                    current_y = ys[i]

                    norm_x = (current_x - min_x_val) / width if width > 0 else 0.5
                    norm_y = (current_y - min_y_val) / height if height > 0 else 0.5

                    sequence.append(
                        [
                            t_offset,
                            norm_x,
                            norm_y,
                            forces[i],
                            altitudes[i],
                            azimuths_sin[i],
                            azimuths_cos[i],
                        ]
                    )

                num_points = len(timestamps)
                duration = timestamps[-1] - first_timestamp if num_points > 0 else 0.0

                avg_force, min_force, max_force = calculate_stats(forces)
                avg_altitude, min_altitude, max_altitude = calculate_stats(altitudes)
                avg_azimuth_sin, min_azimuth_sin, max_azimuth_sin = calculate_stats(
                    azimuths_sin
                )
                avg_azimuth_cos, min_azimuth_cos, max_azimuth_cos = calculate_stats(
                    azimuths_cos
                )

                preview_image = generate_preview_image(sequence)

                processed_records.append(
                    {
                        "created_at": created_at,
                        "sequence": sequence,
                        "width": width,
                        "height": height,
                        "preview_image": preview_image,
                        "num_points": num_points,
                        "duration": duration,
                        "avg_force": avg_force,
                        "min_force": min_force,
                        "max_force": max_force,
                        "avg_altitude": avg_altitude,
                        "min_altitude": min_altitude,
                        "max_altitude": max_altitude,
                        "avg_azimuth_sin": avg_azimuth_sin,
                        "min_azimuth_sin": min_azimuth_sin,
                        "max_azimuth_sin": max_azimuth_sin,
                        "avg_azimuth_cos": avg_azimuth_cos,
                        "min_azimuth_cos": min_azimuth_cos,
                        "max_azimuth_cos": max_azimuth_cos,
                        "signer": signer,
                    }
                )

    if not processed_records:
        print("No records processed. Returning an empty dataset.")

    dataset = Dataset.from_list(processed_records)
    dataset_dict = DatasetDict({"train": dataset})

    return dataset_dict
