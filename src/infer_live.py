from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import List

import cv2
import numpy as np
import tensorflow as tf

from dataset_utils import compute_timesteps, load_gestures_yaml
from feature_extractor import HandsFeatureExtractor


def run_keras(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    return model.predict(x[None, ...], verbose=0)[0]


def run_tflite(interpreter: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], x[None, ...].astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details["index"])[0]


def safe_label(names: List[str], pred_id: int) -> str:
    if 0 <= pred_id < len(names):
        return names[pred_id]
    return f"class_{pred_id}"


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
    p.add_argument("--stride_frames", type=int, default=1)
    args = p.parse_args()

    names, name_to_id = load_gestures_yaml(args.gestures_yaml)
    none_id = name_to_id.get("NONE", len(names) - 1)
    cli_T = compute_timesteps(args.window_seconds, args.target_fps)

    if args.use_tflite:
        if not Path(args.tflite_path).exists():
            raise FileNotFoundError(args.tflite_path)
        interpreter = tf.lite.Interpreter(model_path=args.tflite_path)
        interpreter.allocate_tensors()
        predictor = lambda x: run_tflite(interpreter, x)
        model_T = int(interpreter.get_input_details()[0]["shape"][1])
        model_num_classes = int(interpreter.get_output_details()[0]["shape"][-1])
    else:
        model = tf.keras.models.load_model(args.model_path)
        predictor = lambda x: run_keras(model, x)
        model_T = int(model.input_shape[1])
        model_num_classes = int(model.output_shape[-1])

    if model_num_classes != len(names):
        raise ValueError(
            f"El modelo produce {model_num_classes} clases, pero gestures.yaml define {len(names)} clases."
        )

    if model_T != cli_T:
        print(
            "[WARN] El T esperado por el modelo no coincide con window_seconds*target_fps. "
            f"Se usará T={model_T} del modelo en lugar de T={cli_T} calculado por CLI."
        )
    T = model_T

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    extractor = HandsFeatureExtractor()
    buf: deque[np.ndarray] = deque(maxlen=T)
    score_hist: deque[np.ndarray] = deque(maxlen=max(1, args.smooth_n))
    show_top3 = args.show_top3
    label, conf = "NONE", 0.0

    next_sample_time = time.time()
    sample_period = 1.0 / args.target_fps
    sample_counter = 0
    last_pred_sample = 0

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

            if now >= next_sample_time:
                buf.append(feat.copy())
                sample_counter += 1
                next_sample_time += sample_period
                if next_sample_time < now:
                    next_sample_time = now + sample_period

            ready_for_pred = len(buf) == T and (sample_counter - last_pred_sample) >= max(1, args.stride_frames)
            if ready_for_pred:
                x = np.array(buf, dtype=np.float32)
                probs = predictor(x)
                score_hist.append(probs)
                probs_smoothed = np.mean(np.stack(score_hist, axis=0), axis=0)
                pred_id = int(np.argmax(probs_smoothed))
                pred_conf = float(probs_smoothed[pred_id])
                if pred_conf < args.threshold:
                    pred_id = none_id
                    pred_conf = float(probs_smoothed[none_id])
                label, conf = safe_label(names, pred_id), pred_conf
                last_pred_sample = sample_counter

            cv2.putText(frame, f"Pred: {label} ({conf:.2f})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2)
            cv2.putText(frame, f"FPS cam: {fps:.1f}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2)
            cv2.putText(frame, f"Window samples: {len(buf)}/{T}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 220, 20), 2)
            cv2.putText(frame, f"Sample FPS: {args.target_fps:.1f} | stride={max(1, args.stride_frames)}", (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 20), 2)
            cv2.putText(frame, "q=exit, t=toggle top3", (10, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 220, 20), 2)

            if show_top3 and score_hist:
                top = np.argsort(score_hist[-1])[-3:][::-1]
                yy = 164
                for i in top:
                    cv2.putText(
                        frame,
                        f"{safe_label(names, int(i))}: {float(score_hist[-1][i]):.2f}",
                        (10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                    )
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