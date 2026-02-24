from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf
import yaml

from feature_extractor import HandsFeatureExtractor, TOTAL_FEATURES


def compute_timesteps(window_seconds: float, target_fps: float) -> int:
    return int(round(window_seconds * target_fps))


def load_gestures(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        g = yaml.safe_load(f)
    id_to_name = {int(k): v for k, v in g["id_to_name"].items()}
    return [id_to_name[i] for i in sorted(id_to_name)]


def run_keras(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    return model.predict(x[None, ...], verbose=0)[0]


def run_tflite(interpreter: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], x[None, ...].astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Inferencia en vivo de gestos LSE")
    p.add_argument("--gestures_yaml", type=str, default="data/gestures.yaml")
    p.add_argument("--model_path", type=str, default="outputs/checkpoints/best.keras")
    p.add_argument("--tflite_path", type=str, default="outputs/tflite/model.tflite")
    p.add_argument("--use_tflite", action="store_true")
    p.add_argument("--window_seconds", type=float, default=1.5)
    p.add_argument("--target_fps", type=float, default=15.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--show_top3", action="store_true")
    p.add_argument("--smooth_n", type=int, default=1)
    args = p.parse_args()

    names = load_gestures(args.gestures_yaml)
    none_id = names.index("NONE") if "NONE" in names else len(names) - 1
    T = compute_timesteps(args.window_seconds, args.target_fps)

    if args.use_tflite:
        if not Path(args.tflite_path).exists():
            raise FileNotFoundError(args.tflite_path)
        interpreter = tf.lite.Interpreter(model_path=args.tflite_path)
        interpreter.allocate_tensors()
        predictor = lambda x: run_tflite(interpreter, x)
    else:
        model = tf.keras.models.load_model(args.model_path)
        predictor = lambda x: run_keras(model, x)

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    extractor = HandsFeatureExtractor()
    buf: deque = deque(maxlen=T)
    score_hist: deque = deque(maxlen=max(1, args.smooth_n))
    show_top3 = args.show_top3
    label, conf = "NONE", 0.0

    prev_t = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            feat, frame = extractor.extract_feature(frame, draw=True)
            buf.append(feat)

            if len(buf) == T:
                x = np.array(buf, dtype=np.float32)
                probs = predictor(x)
                score_hist.append(probs)
                probs_smoothed = np.mean(np.stack(score_hist, axis=0), axis=0)
                pred_id = int(np.argmax(probs_smoothed))
                pred_conf = float(probs_smoothed[pred_id])
                if pred_conf < args.threshold:
                    pred_id = none_id
                    pred_conf = float(probs_smoothed[none_id])
                label, conf = names[pred_id], pred_conf
                buf.clear()

            cv2.putText(frame, f"Pred: {label} ({conf:.2f})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2)
            cv2.putText(frame, f"Window: {len(buf)}/{T}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2)
            cv2.putText(frame, "q=exit, t=toggle top3", (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 20), 2)

            if show_top3 and score_hist:
                top = np.argsort(score_hist[-1])[-3:][::-1]
                yy = 138
                for i in top:
                    cv2.putText(frame, f"{names[i]}: {score_hist[-1][i]:.2f}", (10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
                    yy += 24

            cv2.imshow("LSE Live Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("t"):
                show_top3 = not show_top3
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
