from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split


@dataclass
class SampleRecord:
    path: str
    y: int
    y_name: str
    meta: dict

    def to_json_dict(self) -> dict:
        return {
            "path": self.path,
            "y": int(self.y),
            "y_name": self.y_name,
            "meta": self.meta,
        }


def compute_timesteps(window_seconds: float, target_fps: float) -> int:
    if window_seconds <= 0:
        raise ValueError("window_seconds debe ser > 0")
    if target_fps <= 0:
        raise ValueError("target_fps debe ser > 0")
    return max(1, int(round(window_seconds * target_fps)))


def normalize_manifest_path(path: str | Path) -> Path:
    return Path(str(path).replace("\\", "/"))


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
                    path=str(obj["path"]),
                    y=int(obj["y"]),
                    y_name=obj.get("y_name", ""),
                    meta=obj.get("meta", {}),
                )
            )
    return records


def save_manifest(records: Sequence[SampleRecord], manifest_path: str | Path) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


def load_gestures_yaml(path: str | Path) -> Tuple[List[str], Dict[str, int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe gestures.yaml: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, dict) or "id_to_name" not in obj or "name_to_id" not in obj:
        raise ValueError("gestures.yaml debe contener id_to_name y name_to_id")

    id_to_name = {int(k): str(v) for k, v in obj["id_to_name"].items()}
    names = [id_to_name[i] for i in sorted(id_to_name)]
    name_to_id = {str(k): int(v) for k, v in obj["name_to_id"].items()}

    for idx, name in enumerate(names):
        mapped_id = name_to_id.get(name)
        if mapped_id != idx:
            raise ValueError(
                f"Inconsistencia en gestures.yaml: '{name}' tiene id {mapped_id}, pero en id_to_name ocupa {idx}"
            )
    return names, name_to_id


def summarize_class_counts(records: Sequence[SampleRecord]) -> Dict[int, int]:
    counts = Counter(int(r.y) for r in records)
    return {int(k): int(v) for k, v in sorted(counts.items())}


def validate_records_against_gestures(records: Sequence[SampleRecord], gesture_names: Sequence[str]) -> None:
    if not records:
        raise ValueError("El manifest está vacío.")

    valid_ids = set(range(len(gesture_names)))
    seen_ids = [int(r.y) for r in records]
    invalid_ids = sorted(set(seen_ids) - valid_ids)
    if invalid_ids:
        raise ValueError(f"Hay etiquetas en el manifest fuera de gestures.yaml: {invalid_ids}")

    mismatches = []
    for r in records:
        expected = gesture_names[int(r.y)]
        if r.y_name and r.y_name != expected:
            mismatches.append((r.path, r.y, r.y_name, expected))
    if mismatches:
        example = mismatches[0]
        raise ValueError(
            "Hay muestras con y_name inconsistente respecto a gestures.yaml. "
            f"Ejemplo: path={example[0]!r}, y={example[1]}, y_name_manifest={example[2]!r}, esperado={example[3]!r}"
        )

    counts = summarize_class_counts(records)
    missing_ids = [idx for idx in range(len(gesture_names)) if counts.get(idx, 0) == 0]
    if missing_ids:
        missing_names = [gesture_names[idx] for idx in missing_ids]
        raise ValueError(
            "Hay clases definidas en gestures.yaml sin muestras en el manifest. "
            f"IDs ausentes: {missing_ids}; nombres: {missing_names}."
        )


def resolve_sample_path(record_path: str | Path, data_dir: str | Path) -> Path | None:
    data_dir = Path(data_dir)
    base_path = normalize_manifest_path(record_path)
    candidates = []

    if base_path.is_absolute():
        candidates.append(base_path)
    else:
        candidates.append(data_dir / base_path)
        candidates.append(base_path)
        if base_path.parts and base_path.parts[0] == data_dir.name:
            candidates.append(Path(*base_path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_npz_samples(
    records: Sequence[SampleRecord],
    data_dir: str | Path,
    strict_missing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    xs, ys = [], []
    missing_paths: List[str] = []

    for r in records:
        sample_path = resolve_sample_path(r.path, data_dir)
        if sample_path is None:
            missing_paths.append(str(r.path))
            continue

        with np.load(sample_path, allow_pickle=False) as npz:
            xs.append(npz["X"].astype(np.float32))
            ys.append(int(npz["y"]))

    if missing_paths and strict_missing:
        preview = ", ".join(missing_paths[:5])
        raise FileNotFoundError(
            "No se encontraron algunas muestras del manifest. "
            f"Ejemplos: {preview}"
        )

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
        raise ValueError("train_size + val_size + test_size debe sumar 1.0")

    if len(records) < 3:
        raise ValueError("Se necesitan al menos 3 muestras para dividir train/val/test")

    y = np.array([int(r.y) for r in records], dtype=np.int32)
    counts = Counter(y.tolist())
    low_count = {cls: cnt for cls, cnt in counts.items() if cnt < 3}
    if low_count:
        raise ValueError(
            "No se puede hacer split train/val/test estratificado porque hay clases con menos de 3 muestras: "
            f"{low_count}"
        )

    idx = np.arange(len(records))
    train_idx, tmp_idx = train_test_split(
        idx,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=y,
    )

    y_tmp = y[tmp_idx]
    val_ratio_over_tmp = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=(1.0 - val_ratio_over_tmp),
        random_state=seed,
        stratify=y_tmp,
    )

    as_list = lambda indices: [records[int(i)] for i in indices]
    return as_list(train_idx), as_list(val_idx), as_list(test_idx)


def compute_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights: Dict[int, float] = {}
    for c in range(num_classes):
        weights[c] = float(total / (num_classes * max(counts[c], 1.0)))
    return weights


def _augment_numpy(
    x: np.ndarray,
    jitter_std: float = 0.01,
    temporal_dropout_prob: float = 0.05,
) -> np.ndarray:
    x_aug = x.copy()

    noise = np.random.normal(0.0, jitter_std, size=x_aug[:, :126].shape).astype(np.float32)
    x_aug[:, :126] += noise

    if temporal_dropout_prob > 0:
        frame_keep = (np.random.rand(x_aug.shape[0]) > temporal_dropout_prob).astype(np.float32)
        x_aug *= frame_keep[:, None]

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

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)