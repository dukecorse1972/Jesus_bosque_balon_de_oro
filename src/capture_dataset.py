from __future__ import annotations

import argparse
import re
import string
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

from feature_extractor import HandsFeatureExtractor, TOTAL_FEATURES
from dataset_utils import compute_timesteps


RESERVED_KEYS = {"n", "s", "r", "x", "m", "c", " "}


def ensure_gestures_yaml(path: Path) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    names = [f"gesture_{i:02d}" for i in range(1, 21)] + ["NONE"]
    obj = {
        "id_to_name": {i: n for i, n in enumerate(names)},
        "name_to_id": {n: i for i, n in enumerate(names)},
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    return obj


def default_keymap(names: List[str]) -> Dict[str, int]:
    keys = [k for k in (list("1234567890") + list(string.ascii_lowercase)) if k not in RESERVED_KEYS]
    mapping: Dict[str, int] = {}
    for idx in range(len(names) - 1):
        if idx >= len(keys):
            break
        mapping[keys[idx]] = idx
    mapping["n"] = len(names) - 1
    return mapping


def safe_class_dirname(y_id: int, y_name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", y_name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_") or "class"
    return f"{y_id:02d}_{clean}"


def draw_overlay(frame: np.ndarray, lines: List[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 255, 30), 2, cv2.LINE_AA)
        y += 24


def draw_big_text_center(frame: np.ndarray, text: str, font_scale: float = 5.0) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, int(round(font_scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(frame, text, (x + 8, y + 8), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (30, 255, 30), thickness, cv2.LINE_AA)


def draw_text_top_center(frame: np.ndarray, text: str, y: int, font_scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(font_scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    _, w = frame.shape[:2]
    x = (w - tw) // 2
    cv2.putText(frame, text, (x + 3, y + 3), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (30, 255, 30), thickness, cv2.LINE_AA)


def draw_rec_bottom_right(frame: np.ndarray) -> None:
    text = "REC"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]
    x, y = w - tw - 20, h - 20
    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (30, 255, 30), thickness, cv2.LINE_AA)


def save_sample(
    x_seq: np.ndarray,
    y_id: int,
    y_name: str,
    out_dir: Path,
) -> Path:
    """Guarda la muestra .npz en la carpeta de su clase. Sin manifest."""
    class_dir = out_dir / safe_class_dirname(y_id, y_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = class_dir / f"sample_{ts}.npz"
    np.savez_compressed(
        out_path,
        X=x_seq.astype(np.float32),
        y=np.int32(y_id),
        y_name=np.array(y_name),
    )
    return out_path


def resample_sequence_nearest(
    features: List[np.ndarray],
    timestamps: List[float],
    target_fps: float,
    total_steps: int,
) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError("total_steps debe ser > 0")
    if not features:
        return np.zeros((total_steps, TOTAL_FEATURES), dtype=np.float32)

    feats = np.stack(features, axis=0).astype(np.float32)
    ts = np.asarray(timestamps, dtype=np.float32)
    sample_times = (np.arange(total_steps, dtype=np.float32) / float(target_fps)).astype(np.float32)

    right_idx = np.searchsorted(ts, sample_times, side="left")
    right_idx = np.clip(right_idx, 0, len(ts) - 1)
    left_idx  = np.clip(right_idx - 1, 0, len(ts) - 1)

    left_dist  = np.abs(sample_times - ts[left_idx])
    right_dist = np.abs(ts[right_idx] - sample_times)
    final_idx  = np.where(right_dist < left_dist, right_idx, left_idx)
    return feats[final_idx]


def start_recording(now: float) -> Tuple[str, float, List[np.ndarray], List[float]]:
    return "RECORDING", now, [], []


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura dataset LSE aislado (MediaPipe Hands)")
    parser.add_argument("--data_dir",            type=str,   default="data")
    parser.add_argument("--raw_subdir",          type=str,   default="raw")
    parser.add_argument("--gestures_yaml",       type=str,   default="gestures.yaml")
    parser.add_argument("--window_seconds",      type=float, default=1.5)
    parser.add_argument("--target_fps",          type=float, default=15.0)
    parser.add_argument("--camera_id",           type=int,   default=0)
    parser.add_argument("--countdown_seconds",   type=int,   default=3)
    parser.add_argument("--auto_period_seconds", type=float, default=3.0)
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    raw_dir       = data_dir / args.raw_subdir
    gestures_path = data_dir / args.gestures_yaml
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    gestures   = ensure_gestures_yaml(gestures_path)
    id_to_name = {int(k): str(v) for k, v in gestures["id_to_name"].items()}
    names      = [id_to_name[i] for i in sorted(id_to_name.keys())]
    keymap     = default_keymap(names)

    print("=== Mapa de teclas ===")
    for k, idx in keymap.items():
        print(f"  [{k}] -> {names[idx]} (id={idx})")
    print(
        "SPACE=grabar 1 muestra | c/m=modo auto ON/OFF | "
        "r=repetir | s=contadores | x/ESC=salir"
    )

    T   = compute_timesteps(args.window_seconds, args.target_fps)
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    extractor  = HandsFeatureExtractor()
    selected_id = 0
    state       = "READY"
    countdown_end = 0.0
    rec_start     = 0.0
    recording_features: List[np.ndarray] = []
    recording_times:    List[float]      = []
    saved_counter = Counter()

    auto_mode      = False
    auto_next_start = 0.0
    auto_first     = True

    prev_time = time.time()
    fps       = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            dt  = max(now - prev_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            feat, frame = extractor.extract_feature(frame, draw=True)

            if auto_mode and state == "READY" and now >= auto_next_start:
                if auto_first:
                    state = "COUNTDOWN"
                    countdown_end = now + args.countdown_seconds
                    auto_first = False
                else:
                    state, rec_start, recording_features, recording_times = start_recording(now)

            if state == "COUNTDOWN" and now >= countdown_end:
                state, rec_start, recording_features, recording_times = start_recording(now)

            if state == "RECORDING":
                recording_features.append(feat.copy())
                recording_times.append(float(now - rec_start))
                elapsed = now - rec_start

                if elapsed >= args.window_seconds:
                    seq    = resample_sequence_nearest(recording_features, recording_times, args.target_fps, T)
                    y_name = names[selected_id]
                    save_sample(seq, selected_id, y_name, raw_dir)
                    saved_counter[y_name] += 1
                    state = "READY"
                    recording_features = []
                    recording_times    = []

                    if auto_mode:
                        pause = max(0.0, args.auto_period_seconds - args.window_seconds)
                        auto_next_start = now + pause

            if auto_mode:
                total  = sum(saved_counter.values())
                this_g = saved_counter[names[selected_id]]
                draw_text_top_center(frame, f"STATE: {state}", y=45, font_scale=1.0)
                draw_text_top_center(frame, f"TOTAL: {total}   |   THIS: {this_g}", y=90, font_scale=1.6)
                if state == "READY":
                    left_wait = max(0.0, auto_next_start - now)
                    draw_text_top_center(frame, f"NEXT REC IN: {left_wait:.1f}s", y=130, font_scale=0.9)
                if state == "COUNTDOWN":
                    left = max(1, int(round(countdown_end - now))) if (countdown_end - now) > 0 else 0
                    draw_big_text_center(frame, str(left), font_scale=5.0)
                if state == "RECORDING":
                    draw_rec_bottom_right(frame)
            else:
                lines = [
                    f"Gesture: {names[selected_id]} (id={selected_id})",
                    f"State: {state}",
                    f"Auto mode: {'ON' if auto_mode else 'OFF'}",
                    f"Auto period: {args.auto_period_seconds:.1f}s",
                    f"Saved total: {sum(saved_counter.values())}",
                    f"Saved this gesture: {saved_counter[names[selected_id]]}",
                    f"FPS: {fps:.1f}",
                    f"Window: {args.window_seconds:.2f}s @ {args.target_fps:.1f} FPS -> T={T}, F={TOTAL_FEATURES}",
                ]
                if state == "COUNTDOWN":
                    left = max(0, int(round(countdown_end - now)))
                    lines.append(f"Starting in: {left}")
                if state == "RECORDING":
                    lines.append(f"Captured camera frames: {len(recording_features)}")
                if state == "READY" and auto_mode:
                    lines.append(f"Next auto record in: {max(0.0, auto_next_start - now):.1f}s")
                draw_overlay(frame, lines)

            cv2.imshow("LSE Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("x")):
                break
            if key == ord(" ") and state == "READY":
                state = "COUNTDOWN"
                countdown_end = time.time() + args.countdown_seconds
            elif key in (ord("m"), ord("c")):
                auto_mode = not auto_mode
                if auto_mode:
                    auto_first      = True
                    auto_next_start = time.time()
                    print(f"Modo automático ACTIVADO (periodo={args.auto_period_seconds}s).")
                else:
                    print("Modo automático DESACTIVADO.")
            elif key == ord("s"):
                print("--- Contadores en sesión ---")
                for n in names:
                    if saved_counter[n] > 0:
                        print(f"{n}: {saved_counter[n]}")
            elif key == ord("r"):
                state = "READY"
                recording_features = []
                recording_times    = []
                print("Estado reiniciado.")
            else:
                kchar = chr(key).lower() if 0 <= key < 256 else ""
                if kchar in keymap:
                    selected_id = keymap[kchar]
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()