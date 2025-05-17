import io
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image as PilImage
from torch.utils.data import Sampler
from torch.utils.data.dataloader import default_collate


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


def transform_sequence_from_discrete_to_continuous(
    dataset: DatasetDict,
) -> DatasetDict:
    """
    Transform discrete sequence data to 240Hz regularly sampled sequences using linear interpolation.
    If the gap between points is > 1/120s, fill with zeros.
    """

    def _transform_sequence(record):
        sequence = record["sequence"]
        duration = record["duration"]
        if not sequence or duration <= 0:
            record["sequence"] = []
            return record
        # Output: regularly sampled at 240Hz
        step = 1.0 / 240
        num_steps = int(math.ceil(duration / step)) + 1
        feature_dim = len(sequence[0])
        new_sequence = []
        seq_idx = 0
        for i in range(num_steps):
            t = i * step
            # Find the two points in sequence that t falls between
            while seq_idx + 1 < len(sequence) and sequence[seq_idx + 1][0] < t:
                seq_idx += 1
            if seq_idx + 1 < len(sequence):
                t0, t1 = sequence[seq_idx][0], sequence[seq_idx + 1][0]
                p0, p1 = sequence[seq_idx], sequence[seq_idx + 1]
                if t1 - t0 > 1 / 120:
                    # Large gap, fill with zeros
                    new_sequence.append([0.0] * feature_dim)
                else:
                    # Linear interpolation
                    alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                    interp = [
                        p0[j] + (p1[j] - p0[j]) * alpha for j in range(feature_dim)
                    ]
                    new_sequence.append(interp)
            else:
                # After last point, fill with zeros
                new_sequence.append([0.0] * feature_dim)
        record["sequence"] = new_sequence
        return record

    dataset_dict = dataset.map(_transform_sequence)
    return dataset_dict


def prepare_samples(
    dataset: Dataset,
    num_references: int = 1,
    max_samples_per_signer: int = 200,
) -> Dataset:
    """
    Prepare pairs of reference and target sequences for training.
    Each sample consists of N reference sequences and a target sequence from the same signer.
    Uses all possible combinations (C(count, N+1)) for each signer, up to max_samples_per_signer, using reservoir sampling for memory efficiency.
    """
    # Group records by signer
    signer_records = {}
    for record in dataset:
        signer = record["signer"]
        if signer not in signer_records:
            signer_records[signer] = []
        signer_records[signer].append(record)

    new_samples = []
    for signer, records in signer_records.items():
        if len(records) < num_references + 1:
            print(f"Signer {signer} has only {len(records)} samples, skipping.")
            continue
        # Reservoir sampling for combinations
        reservoir = []
        for i, combo in enumerate(combinations(records, num_references + 1)):
            if i < max_samples_per_signer:
                reservoir.append(combo)
            else:
                j = random.randint(0, i)
                if j < max_samples_per_signer:
                    reservoir[j] = combo
        for combo in reservoir:
            ref_records = combo[:num_references]
            target_record = combo[num_references]
            new_samples.append(
                {
                    "references": [r["sequence"] for r in ref_records],
                    "reference_widths": [r["width"] for r in ref_records],
                    "reference_heights": [r["height"] for r in ref_records],
                    "target": target_record["sequence"],
                    "target_width": target_record["width"],
                    "target_height": target_record["height"],
                    "signer": signer,
                }
            )
    return Dataset.from_list(new_samples)


def load_default_dataset() -> DatasetDict:
    """
    ## Splits
    - train: ~18.6k samples (6 signer)
    - test: ~3.2k samples (1 signer)
    ## Features
    - references (list of list of list of float): A list of N reference sequences from the same signer.
      Each reference sequence is processed (time removed, padded/truncated). shape: (V=1~16, T=4096, F=6)
    - reference_widths (list of float): A list of widths corresponding to each reference sequence. shape: (V=1~16)
    - reference_heights (list of float): A list of heights corresponding to each reference sequence. shape: (V=1~16)
    - target (list of list of float): The target sequence from the same signer, processed similarly to references. shape: (T=4096, F=6)
    - target_width (float): The width corresponding to the target sequence.
    - target_height (float): The height corresponding to the target sequence.
    - signer (string): The identifier for the signer (same for references and target in a sample).
    """
    return load_dataset("JacobLinCool/SignSeq-dataset-1x16")


class VariableVBatchSampler(Sampler):
    """
    A batch sampler that groups samples by the number of references (V)
    and ensures each batch contains samples with the same V.
    It also handles shuffling of V-groups and samples within V-groups.
    """

    def __init__(
        self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.v_groups = self._group_by_v()
        self.v_keys = list(self.v_groups.keys())

        self.num_batches = 0
        for v_key in self.v_keys:
            indices_for_v = self.v_groups[v_key]
            if self.drop_last:
                self.num_batches += len(indices_for_v) // self.batch_size
            else:
                self.num_batches += math.ceil(len(indices_for_v) / self.batch_size)

    def _group_by_v(self):
        v_groups = defaultdict(list)
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            # V is the number of reference sequences
            v_value = len(sample["references"])
            v_groups[v_value].append(i)
        return v_groups

    def __iter__(self):
        all_batches = []

        v_keys_iter = list(self.v_keys)
        if self.shuffle:
            random.shuffle(v_keys_iter)

        for v_val in v_keys_iter:
            indices_for_v = list(self.v_groups[v_val])
            if self.shuffle:
                random.shuffle(indices_for_v)

            for i in range(0, len(indices_for_v), self.batch_size):
                batch_indices = indices_for_v[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch_indices)

        if (
            self.shuffle
        ):  # Shuffle the order of all generated batches from different V-groups
            random.shuffle(all_batches)

        return iter(all_batches)

    def __len__(self):
        return self.num_batches


def collate_signseq(batch_list):
    """
    Custom collate function for SignSeq data.
    Assumes all samples in batch_list have the same V (number of references),
    ensured by VariableVBatchSampler.
    Constructs 'ref_controls' and 'control' tensors.
    """
    # Pop non-tensor items or items that default_collate might struggle with if they were variable.
    # 'signer' is a list of strings, handle separately.
    signers = [s.pop("signer") for s in batch_list if "signer" in s]

    # Use default_collate for the rest of the tensor features
    # (e.g., 'references', 'reference_widths', 'reference_heights', 'target', 'target_width', 'target_height')
    collated_batch = default_collate(batch_list)

    # Restore signers if they were present
    if signers:
        collated_batch["signer"] = signers

    # Construct 'ref_controls' and 'control' which are expected by the model
    # Ensure the keys exist from the dataset features
    if "reference_widths" in collated_batch and "reference_heights" in collated_batch:
        ref_widths = collated_batch["reference_widths"]
        ref_heights = collated_batch["reference_heights"]
        collated_batch["ref_controls"] = torch.stack((ref_widths, ref_heights), dim=-1)

    if "target_width" in collated_batch and "target_height" in collated_batch:
        target_width = collated_batch["target_width"]
        target_height = collated_batch["target_height"]
        collated_batch["control"] = torch.stack((target_width, target_height), dim=-1)

    # Rename 'references' from dataset to 'refs' for the model input
    if "references" in collated_batch:
        collated_batch["refs"] = collated_batch.pop("references")

    return collated_batch
