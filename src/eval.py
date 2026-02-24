from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score

from dataset_utils import load_manifest, load_npz_samples, make_tf_dataset, stratified_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluación de modelo TCN")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--manifest", type=str, default="manifest.jsonl")
    p.add_argument("--model_path", type=str, default="outputs/checkpoints/best.keras")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gestures_yaml", type=str, default="data/gestures.yaml")
    p.add_argument("--save_cm_png", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = load_manifest(Path(args.data_dir) / args.manifest)
    _, _, test_records = stratified_split(records, seed=args.seed)
    x_test, y_test = load_npz_samples(test_records, args.data_dir)

    model = tf.keras.models.load_model(args.model_path)
    ds = make_tf_dataset(x_test, y_test, args.batch_size, training=False)
    probs = model.predict(ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    with open(args.gestures_yaml, "r", encoding="utf-8") as f:
        g = yaml.safe_load(f)
    id_to_name = {int(k): v for k, v in g["id_to_name"].items()}
    none_id = int(g["name_to_id"].get("NONE", max(id_to_name)))

    f1_none = f1_score((y_test == none_id).astype(int), (y_pred == none_id).astype(int), zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(id_to_name))

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro:.4f}")
    print(f"F1(NONE): {f1_none:.4f}")
    print("Confusion matrix:")
    print(cm)

    if args.save_cm_png:
        labels = [id_to_name[i] for i in sorted(id_to_name)]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12, 12))
        disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
        fig.tight_layout()
        out_path = Path("outputs") / "confusion_matrix.png"
        fig.savefig(out_path, dpi=180)
        print(f"Guardado: {out_path}")


if __name__ == "__main__":
    main()
