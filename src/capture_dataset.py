from __future__ import annotations

import argparse
import json
import re
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
    """
    Texto centrado arriba con sombra, tamaño configurable.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(font_scale * 2)))

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]
    x = (w - tw) // 2

    cv2.putText(frame, text, (x + 3, y + 3), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (30, 255, 30), thickness, cv2.LINE_AA)


def draw_rec_bottom_right(frame: np.ndarray) -> None:
    """
    REC pequeño en esquina inferior derecha.
    """
    text = "REC"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    h, w = frame.shape[:2]
    x = w - tw - 20
    y = h - 20

    cv2.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, (30, 255, 30), thickness, cv2.LINE_AA)


def save_sample(
    x_seq: np.ndarray,
    y_id: int,
    y_name: str,
    out_dir: Path,
    manifest_path: Path,
    data_dir: Path,
    meta: dict,
) -> Path:
    class_dir = out_dir / safe_class_dirname(y_id, y_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_name = f"sample_{ts}.npz"
    out_path = class_dir / file_name
    np.savez_compressed(
        out_path,
        X=x_seq.astype(np.float32),
        y=np.int32(y_id),
        y_name=np.array(y_name),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )

    rel_path = out_path.resolve().relative_to(data_dir.resolve())
    line = {
        "path": str(rel_path).replace("\\", "/"),
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

    # Auto
    parser.add_argument("--auto_period_seconds", type=float, default=2.0)
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
        "SPACE=grabar 1 muestra | c/m=modo auto ON/OFF (solo 1er countdown) | "
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
    auto_first = True  # solo la primera vez se hace countdown

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

            # Auto trigger:
            # - Primera vez: COUNTDOWN
            # - Siguientes: RECORDING directo (sin countdown)
            if auto_mode and state == "READY" and now >= auto_next_start:
                if auto_first:
                    state = "COUNTDOWN"
                    countdown_end = now + args.countdown_seconds
                    auto_first = False
                else:
                    state = "RECORDING"
                    rec_start = now
                    frame_buffer.clear()

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

                    y_name = names[selected_id]
                    meta = {
                        "timestamp": datetime.now().isoformat(),
                        "fps_cam": float(fps),
                        "window_seconds": float(args.window_seconds),
                        "target_fps": float(args.target_fps),
                        "subject": args.subject,
                        "lighting_note": args.lighting_note,
                        "auto_mode": bool(auto_mode),
                        "auto_period_seconds": float(args.auto_period_seconds),
                        "storage_subdir": f"{args.raw_subdir}/{safe_class_dirname(selected_id, y_name)}",
                    }
                    save_sample(seq, selected_id, y_name, raw_dir, manifest_path, data_dir, meta)
                    saved_counter[y_name] += 1
                    state = "READY"

                    # Programa siguiente inicio en modo auto:
                    if auto_mode:
                        pause = max(0.0, args.auto_period_seconds - args.window_seconds)
                        auto_next_start = now + pause

            # --- OVERLAY ---
            if auto_mode:
                total = sum(saved_counter.values())
                this_g = saved_counter[names[selected_id]]

                # Todo pequeño excepto la cuenta atrás (y contadores grandes)
                draw_text_top_center(frame, f"STATE: {state}", y=45, font_scale=1.0)

                # Contadores GRANDES
                draw_text_top_center(frame, f"TOTAL: {total}   |   THIS: {this_g}", y=90, font_scale=1.6)

                # En descanso (READY en auto), mostrar cuánto falta (pequeño)
                if state == "READY":
                    left_wait = max(0.0, auto_next_start - now)
                    draw_text_top_center(frame, f"NEXT REC IN: {left_wait:.1f}s", y=130, font_scale=0.9)

                # Cuenta atrás GRANDE en el centro (no tocar)
                if state == "COUNTDOWN":
                    left = max(0, int(round(countdown_end - now)))
                    left = max(1, left) if (countdown_end - now) > 0 else 0
                    draw_big_text_center(frame, str(left), font_scale=5.0)

                # REC pequeño abajo derecha
                if state == "RECORDING":
                    draw_rec_bottom_right(frame)

            else:
                # Manual: overlay completo como antes
                lines = [
                    f"Gesture: {names[selected_id]} (id={selected_id})",
                    f"State: {state}",
                    f"Auto mode: {'ON' if auto_mode else 'OFF'}",
                    f"Auto period: {args.auto_period_seconds:.1f}s",
                    f"Saved total: {sum(saved_counter.values())}",
                    f"Saved this gesture: {saved_counter[names[selected_id]]}",
                    f"FPS: {fps:.1f}",
                    f"Window T={T}, F={TOTAL_FEATURES}",
                ]
                if state == "COUNTDOWN":
                    left = max(0, int(round(countdown_end - now)))
                    lines.append(f"Starting in: {left}")
                if state == "READY":
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
                    auto_first = True
                    auto_next_start = time.time()
                    print(
                        f"Modo automático ACTIVADO (periodo={args.auto_period_seconds}s, "
                        f"countdown inicial={args.countdown_seconds}s)."
                    )
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