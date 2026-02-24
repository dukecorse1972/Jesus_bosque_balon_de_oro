from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


@dataclass
class SampleRecord:
    path: str
    y: int
    y_name: str
    meta: dict


def compute_timesteps(window_seconds: float, target_fps: float) -> int:
    return int(round(window_seconds * target_fps))


def load_manifest(manifest_path: str | Path) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return records
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            records.append(
                SampleRecord(
                    path=obj["path"],
                    y=int(obj["y"]),
                    y_name=obj.get("y_name", ""),
                    meta=obj.get("meta", {}),
                )
            )
    return records


def load_npz_samples(records: Sequence[SampleRecord], data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    xs, ys = [], []
    for r in records:
        base_path = Path(r.path)
        candidates = [base_path]
        if not base_path.is_absolute():
            candidates.append(data_dir / base_path)
            # Compatibilidad con manifests antiguos que guardaban "data/raw/..."
            if base_path.parts and base_path.parts[0] == data_dir.name:
                candidates.append(Path(*base_path.parts[1:]))

        sample_path = next((c for c in candidates if c.exists()), None)
        if sample_path is None:
            continue

        with np.load(sample_path, allow_pickle=False) as npz:
            xs.append(npz["X"].astype(np.float32))
            ys.append(int(npz["y"]))
    if not xs:
        raise ValueError("No se pudieron cargar muestras desde el manifest.")
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int32)


def stratified_split(
    records: Sequence[SampleRecord],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SampleRecord], List[SampleRecord], List[SampleRecord]]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train+val+test debe sumar 1.0")
    y = np.array([r.y for r in records])
    idx = np.arange(len(records))

    train_idx, tmp_idx = train_test_split(
        idx, test_size=(1.0 - train_size), random_state=seed, stratify=y
    )
    y_tmp = y[tmp_idx]
    val_ratio_over_tmp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=(1.0 - val_ratio_over_tmp), random_state=seed, stratify=y_tmp
    )
    as_list = lambda indices: [records[i] for i in indices]
    return as_list(train_idx), as_list(val_idx), as_list(test_idx)


def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = {}
    for c in range(num_classes):
        weights[c] = float(total / (num_classes * max(counts[c], 1.0)))
    return weights


def _augment_numpy(
    x: np.ndarray,
    jitter_std: float = 0.01,
    temporal_dropout_prob: float = 0.05,
) -> np.ndarray:
    x_aug = x.copy()
    # landmarks: first 126 dims (63 left + 63 right)
    noise = np.random.normal(0.0, jitter_std, size=x_aug[:, :126].shape).astype(np.float32)
    x_aug[:, :126] += noise

    if temporal_dropout_prob > 0:
        frame_keep = (np.random.rand(x_aug.shape[0]) > temporal_dropout_prob).astype(np.float32)
        frame_keep = frame_keep[:, None]
        x_aug *= frame_keep
    return x_aug.astype(np.float32)


def make_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    training: bool,
    augment: bool = False,
    shuffle_buffer: int = 2048,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(x)), reshuffle_each_iteration=True)

    if training and augment:
        def _tf_aug(xb: tf.Tensor, yb: tf.Tensor):
            xa = tf.numpy_function(_augment_numpy, [xb], Tout=tf.float32)
            xa.set_shape(xb.shape)
            return xa, yb

        ds = ds.map(_tf_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
