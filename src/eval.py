from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_utils import (
    load_gestures_yaml,
    load_manifest,
    load_npz_samples,
    make_tf_dataset,
    validate_records_against_gestures,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluación del modelo TCN para gestos LSE")
    p.add_argument("--data_dir", type=str, default="data",
                    help="Directorio raíz con los datos (.npz)")
    p.add_argument("--test_split", type=str, required=True,
                    help="Ruta al manifest del split de test (.jsonl)")
    p.add_argument("--model_path", type=str, required=True,
                    help="Ruta al modelo entrenado (.keras)")
    p.add_argument("--gestures_yaml", type=str, default="data/gestures.yaml",
                    help="Ruta al archivo gestures.yaml")
    p.add_argument("--batch_size", type=int, default=32,
                    help="Tamaño de batch para predicción")
    p.add_argument("--save_cm_png", action="store_true",
                    help="Guardar la matriz de confusión como imagen PNG")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    test_split_path = Path(args.test_split)
    model_path = Path(args.model_path)
    gestures_path = Path(args.gestures_yaml)

    # --- Cargar nombres de gestos ---
    gesture_names, _ = load_gestures_yaml(gestures_path)
    num_classes = len(gesture_names)

    # --- Cargar manifest de test ---
    records = load_manifest(test_split_path)
    if not records:
        raise SystemExit(f"Error: el manifest está vacío o no existe: {test_split_path}")

    validate_records_against_gestures(records, gesture_names)

    print(f"Cargando {len(records)} muestras de test...")
    x_test, y_test = load_npz_samples(records, data_dir, strict_missing=True)

    test_ds = make_tf_dataset(x_test, y_test, args.batch_size, training=False)

    # --- Cargar modelo ---
    print(f"Cargando modelo desde {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # --- Predicciones ---
    print("Ejecutando predicciones...")
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # --- Métricas globales ---
    all_labels = list(range(num_classes))
    test_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=all_labels, zero_division=0)

    print("\n" + "=" * 50)
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test Macro-F1 : {macro_f1:.4f}")
    print("=" * 50 + "\n")

    # --- Reporte por clase ---
    report = classification_report(
        y_test, y_pred,
        labels=all_labels,
        target_names=gesture_names,
        digits=4,
        zero_division=0,
    )
    print("Classification Report:\n")
    print(report)

    # --- Matriz de confusión ---
    if args.save_cm_png:
        out_dir = (
            model_path.parent.parent
            if model_path.parent.name == "checkpoints"
            else model_path.parent
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cm_path = out_dir / "confusion_matrix.png"

        print("Generando matriz de confusión...")
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)

        fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes * 0.8)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gesture_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", ax=ax)
        plt.title("Matriz de Confusión — Test Split")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Matriz de confusión guardada en: {cm_path}")


if __name__ == "__main__":
    main()