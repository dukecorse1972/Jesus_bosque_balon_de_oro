from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_utils import load_gestures_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta Keras -> SavedModel + TFLite")
    p.add_argument("--model_path", type=str, default="outputs/checkpoints/best.keras")
    p.add_argument("--gestures_yaml", type=str, default="data/gestures.yaml")
    p.add_argument("--saved_model_dir", type=str, default="outputs/saved_model")
    p.add_argument("--tflite_path", type=str, default="outputs/tflite/model.tflite")
    p.add_argument("--labels_path", type=str, default="outputs/tflite/labels.txt")
    p.add_argument("--quantize_dynamic", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify_samples", type=int, default=3)
    return p.parse_args()


def export_saved_model(model: tf.keras.Model, saved_dir: Path) -> None:
    saved_dir.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "export"):
        model.export(str(saved_dir))
    else:
        tf.saved_model.save(model, str(saved_dir))


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)

    gesture_names, _ = load_gestures_yaml(args.gestures_yaml)
    model_num_classes = int(model.output_shape[-1])
    if model_num_classes != len(gesture_names):
        raise ValueError(
            f"El modelo produce {model_num_classes} clases, pero gestures.yaml define {len(gesture_names)} clases."
        )

    saved_dir = Path(args.saved_model_dir)
    export_saved_model(model, saved_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if args.quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = Path(args.tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    labels_path = Path(args.labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(gesture_names) + "\n", encoding="utf-8")

    print(f"SavedModel: {saved_dir}")
    print(f"TFLite: {tflite_path}")
    print(f"Labels: {labels_path}")

    if args.verify:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        input_shape = tuple(int(x) for x in input_details["shape"][1:])
        diffs = []

        for _ in range(max(1, args.verify_samples)):
            x = np.random.randn(*input_shape).astype(np.float32)
            y_keras = model.predict(x[None, ...], verbose=0)[0]
            interpreter.set_tensor(input_details["index"], x[None, ...].astype(np.float32))
            interpreter.invoke()
            y_tflite = interpreter.get_tensor(output_details["index"])[0]
            diffs.append(float(np.max(np.abs(y_keras - y_tflite))))

        print(f"Verificación TFLite: max_abs_diff={max(diffs):.8f}, mean_abs_diff={np.mean(diffs):.8f}")


if __name__ == "__main__":
    main()