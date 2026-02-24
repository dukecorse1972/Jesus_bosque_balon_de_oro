from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta Keras -> SavedModel + TFLite")
    p.add_argument("--model_path", type=str, default="outputs/checkpoints/best.keras")
    p.add_argument("--saved_model_dir", type=str, default="outputs/saved_model")
    p.add_argument("--tflite_path", type=str, default="outputs/tflite/model.tflite")
    p.add_argument("--quantize_dynamic", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)

    saved_dir = Path(args.saved_model_dir)
    saved_dir.parent.mkdir(parents=True, exist_ok=True)
    model.export(str(saved_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_dir))
    if args.quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = Path(args.tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)

    print(f"SavedModel: {saved_dir}")
    print(f"TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
