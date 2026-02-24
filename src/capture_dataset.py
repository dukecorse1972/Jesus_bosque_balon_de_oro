from __future__ import annotations

import argparse
import json
import string
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from feature_extractor import HandsFeatureExtractor, TOTAL_FEATURES


RESERVED_KEYS = {"n", "s", "r", "x", "m", "c", " "}


def compute_timesteps(window_seconds: float, target_fps: float) -> int:
    return int(round(window_seconds * target_fps))


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


def draw_overlay(frame: np.ndarray, lines: List[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 255, 30), 2, cv2.LINE_AA)
        y += 24


def save_sample(
    x_seq: np.ndarray,
    y_id: int,
    y_name: str,
    out_dir: Path,
    manifest_path: Path,
    meta: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"sample_{ts}_y{y_id:02d}.npz"
    out_path = out_dir / file_name
    np.savez_compressed(
        out_path,
        X=x_seq.astype(np.float32),
        y=np.int32(y_id),
        y_name=np.array(y_name),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    line = {
        "path": str(out_path),
        "y": int(y_id),
        "y_name": y_name,
        "meta": meta,
    }
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura dataset LSE aislado (MediaPipe Hands)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--raw_subdir", type=str, default="raw")
    parser.add_argument("--manifest", type=str, default="manifest.jsonl")
    parser.add_argument("--gestures_yaml", type=str, default="gestures.yaml")
    parser.add_argument("--window_seconds", type=float, default=1.5)
    parser.add_argument("--target_fps", type=float, default=15.0)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--countdown_seconds", type=int, default=3)
    parser.add_argument("--auto_pause_seconds", type=float, default=3.0)
    parser.add_argument("--subject", type=str, default="self")
    parser.add_argument("--lighting_note", type=str, default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / args.raw_subdir
    manifest_path = data_dir / args.manifest
    gestures_path = data_dir / args.gestures_yaml
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.touch(exist_ok=True)

    gestures = ensure_gestures_yaml(gestures_path)
    id_to_name = {int(k): v for k, v in gestures["id_to_name"].items()}
    names = [id_to_name[i] for i in sorted(id_to_name.keys())]
    keymap = default_keymap(names)

    print("=== Mapa de teclas ===")
    for k, idx in keymap.items():
        print(f"  [{k}] -> {names[idx]} (id={idx})")
    print(
        "SPACE=grabar 1 muestra | c/m=modo auto ON/OFF (pausa entre muestras) | "
        "r=repetir | s=contadores | x/ESC=salir"
    )

    T = compute_timesteps(args.window_seconds, args.target_fps)
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    extractor = HandsFeatureExtractor()
    selected_id = 0
    state = "READY"
    countdown_end = 0.0
    rec_start = 0.0
    frame_buffer: deque[np.ndarray] = deque(maxlen=T)
    saved_counter = Counter()
    auto_mode = False
    auto_next_start = 0.0

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            dt = max(now - prev_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = now

            feat, frame = extractor.extract_feature(frame, draw=True)

            if auto_mode and state == "READY" and now >= auto_next_start:
                state = "COUNTDOWN"
                countdown_end = now + args.countdown_seconds

            if state == "COUNTDOWN" and now >= countdown_end:
                state = "RECORDING"
                rec_start = now
                frame_buffer.clear()

            if state == "RECORDING":
                frame_buffer.append(feat)
                elapsed = now - rec_start
                if elapsed >= args.window_seconds:
                    seq = np.array(frame_buffer, dtype=np.float32)
                    if len(seq) < T:
                        pad = np.zeros((T - len(seq), TOTAL_FEATURES), dtype=np.float32)
                        seq = np.concatenate([seq, pad], axis=0)
                    elif len(seq) > T:
                        idx = np.linspace(0, len(seq) - 1, T).astype(np.int32)
                        seq = seq[idx]

                    meta = {
                        "timestamp": datetime.now().isoformat(),
                        "fps_cam": float(fps),
                        "window_seconds": float(args.window_seconds),
                        "target_fps": float(args.target_fps),
                        "subject": args.subject,
                        "lighting_note": args.lighting_note,
                        "auto_mode": bool(auto_mode),
                        "auto_pause_seconds": float(args.auto_pause_seconds),
                    }
                    y_name = names[selected_id]
                    save_sample(seq, selected_id, y_name, raw_dir, manifest_path, meta)
                    saved_counter[y_name] += 1
                    state = "READY"
                    if auto_mode:
                        auto_next_start = now + args.auto_pause_seconds

            lines = [
                f"Gesture: {names[selected_id]} (id={selected_id})",
                f"State: {state}",
                f"Auto mode: {'ON' if auto_mode else 'OFF'}",
                f"Saved total: {sum(saved_counter.values())}",
                f"Saved this gesture: {saved_counter[names[selected_id]]}",
                f"FPS: {fps:.1f}",
                f"Window T={T}, F={TOTAL_FEATURES}",
            ]
            if state == "COUNTDOWN":
                left = max(0, int(round(countdown_end - now)))
                lines.append(f"Starting in: {left}")
            if auto_mode and state == "READY":
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
                    auto_next_start = time.time()
                    print(f"Modo automático ACTIVADO (pausa={args.auto_pause_seconds}s).")
                else:
                    print("Modo automático DESACTIVADO.")
            elif key == ord("s"):
                print("--- Contadores en sesión ---")
                for n in names:
                    if saved_counter[n] > 0:
                        print(f"{n}: {saved_counter[n]}")
            elif key == ord("r"):
                state = "READY"
                frame_buffer.clear()
                print("Repetir habilitado: reinicia estado sin guardar.")
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
