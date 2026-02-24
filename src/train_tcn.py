from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from dataset_utils import (
    compute_class_weights,
    load_manifest,
    load_npz_samples,
    make_tf_dataset,
    stratified_split,
)
from metrics import ValMacroF1Callback
from models import build_tcn, model_size_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento TCN para gestos LSE")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--manifest", type=str, default="manifest.jsonl")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window_seconds", type=float, default=1.5)
    p.add_argument("--target_fps", type=float, default=15.0)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--model_size", type=str, default="small", choices=["small", "medium"])
    p.add_argument("--num_classes", type=int, default=21)
    p.add_argument("--outputs_dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / args.manifest
    records = load_manifest(manifest_path)
    if len(records) < args.num_classes:
        raise RuntimeError("Dataset insuficiente para split estratificado.")

    tr_rec, va_rec, te_rec = stratified_split(records, seed=args.seed)
    x_train, y_train = load_npz_samples(tr_rec, data_dir)
    x_val, y_val = load_npz_samples(va_rec, data_dir)
    x_test, y_test = load_npz_samples(te_rec, data_dir)

    train_ds = make_tf_dataset(x_train, y_train, args.batch_size, training=True, augment=args.augment)
    val_ds = make_tf_dataset(x_val, y_val, args.batch_size, training=False)
    test_ds = make_tf_dataset(x_test, y_test, args.batch_size, training=False)

    params = model_size_params(args.model_size)
    model = build_tcn(input_shape=x_train.shape[1:], num_classes=args.num_classes, **params)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    out = Path(args.outputs_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ValMacroF1Callback(val_ds),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(out / "training_log.csv")),
    ]

    class_weight = compute_class_weights(y_train, args.num_classes) if args.use_class_weights else None

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Test macro-F1: {macro_f1:.4f}")

    final_path = ckpt_dir / "last.keras"
    model.save(final_path)

    cfg = vars(args)
    cfg["input_shape"] = list(x_train.shape[1:])
    cfg["split_sizes"] = {"train": len(y_train), "val": len(y_val), "test": len(y_test)}
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
