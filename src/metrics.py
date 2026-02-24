from __future__ import annotations

from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


class ValMacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset: tf.data.Dataset, name: str = "val_macro_f1") -> None:
        super().__init__()
        self.val_dataset = val_dataset
        self.name = name

    def on_epoch_end(self, epoch: int, logs: Dict | None = None) -> None:
        logs = logs or {}
        y_true = []
        y_pred = []
        for xb, yb in self.val_dataset:
            probs = self.model.predict(xb, verbose=0)
            pred = np.argmax(probs, axis=1)
            y_true.append(yb.numpy())
            y_pred.append(pred)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logs[self.name] = float(macro)
        print(f" - {self.name}: {macro:.4f}")
