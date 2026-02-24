# Proyecto LSE (gestos aislados) con TCN + MediaPipe Hands

Proyecto completo en Python para capturar dataset propio, entrenar un modelo TCN en TensorFlow/Keras, evaluar, exportar a TFLite e inferir en vivo desde webcam en PC (Windows), usando **solo manos**.

## Estructura

- `src/capture_dataset.py`: captura de muestras por ventana fija.
- `src/dataset_utils.py`: lectura de manifest, split estratificado, `tf.data`, augment y class weights.
- `src/models.py`: arquitectura TCN causal residual.
- `src/metrics.py`: callback para macro-F1 en validación.
- `src/train_tcn.py`: entrenamiento completo.
- `src/eval.py`: evaluación con accuracy, macro-F1, F1 de NONE y matriz de confusión.
- `src/export_tflite.py`: export a SavedModel + TFLite.
- `src/infer_live.py`: inferencia en vivo Keras o TFLite.
- `data/gestures.yaml`: mapeo editable `id_to_name` y `name_to_id`.
- `data/manifest.jsonl`: registro de muestras guardadas.

## 1) Instalar dependencias

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Capturar dataset (webcam)

```bash
python src/capture_dataset.py --data_dir data --window_seconds 1.5 --target_fps 15 --auto_pause_seconds 3
```

### Teclas de captura

- `1..9`, `0`, `a..z`: seleccionan clases de gestos (`gesture_01`...`gesture_20`) según mapa mostrado en consola.
- `n`: selecciona `NONE`.
- `SPACE`: inicia cuenta atrás y graba **una** secuencia (sin solape).
- `m`: activa/desactiva modo automático (graba secuencias sin parar, con pausa configurable entre muestras).
- `r`: repetir (reinicia estado sin guardar).
- `s`: imprime contadores de sesión por gesto.
- `x` o `ESC`: salir.

### Formato guardado por muestra (`.npz`)

- `X`: `float32`, shape `(T, F)`.
  - `T = round(window_seconds * target_fps)` (por defecto `round(1.5*15)=23`).
  - `F = 130`.
- `y`: `int32`.
- `y_name`: nombre de clase.
- `meta_json`: JSON serializado con metadatos.

### Features por frame (F=130)

- LEFT: `21*(x,y,z)=63` normalizados.
- RIGHT: `63` normalizados.
- `mask_left`, `mask_right` (2).
- `handedness_left`, `handedness_right` (2).

Normalización por mano:
1. Restar `wrist (lm0)`.
2. Escalar por distancia `wrist(0)->middle_mcp(9)`.
3. Sin rotación adicional.

## 3) Entrenar

```bash
python src/train_tcn.py \
  --data_dir data \
  --manifest manifest.jsonl \
  --epochs 40 \
  --batch_size 32 \
  --lr 1e-3 \
  --seed 42 \
  --use_class_weights \
  --augment \
  --model_size small
```

Split por defecto: `70/15/15` estratificado.

Salida principal:
- `outputs/checkpoints/best.keras`
- `outputs/checkpoints/last.keras`
- `outputs/training_log.csv`
- `outputs/config.json`

## 4) Evaluar

```bash
python src/eval.py --data_dir data --manifest manifest.jsonl --model_path outputs/checkpoints/best.keras --save_cm_png
```

Imprime:
- accuracy
- macro-F1
- F1 para clase `NONE`
- matriz de confusión (texto)

Guarda opcional:
- `outputs/confusion_matrix.png`

## 5) Exportar a TFLite

```bash
python src/export_tflite.py --model_path outputs/checkpoints/best.keras
```

Con quantización dinámica opcional:

```bash
python src/export_tflite.py --model_path outputs/checkpoints/best.keras --quantize_dynamic
```

Salidas:
- `outputs/saved_model/`
- `outputs/tflite/model.tflite`

## 6) Inferencia en vivo

### Keras (por defecto, recomendado en PC)

```bash
python src/infer_live.py --model_path outputs/checkpoints/best.keras --gestures_yaml data/gestures.yaml
```

### TFLite

```bash
python src/infer_live.py --use_tflite --tflite_path outputs/tflite/model.tflite --gestures_yaml data/gestures.yaml
```

Flags útiles:
- `--threshold 0.5`: si confianza top-1 < threshold, fuerza `NONE`.
- `--show_top3`: muestra top-3 en overlay.
- `--smooth_n 1`: promedio móvil de predicciones por ventana (poner >1 para suavizar).
- Teclas: `q` salir, `t` toggle top-3.

## Renombrar gestos

Edita `data/gestures.yaml` y cambia nombres en `id_to_name` y `name_to_id` manteniendo consistencia de IDs (0..20). No es necesario cambiar código.

## Notas de robustez

- Si solo se detecta una mano, la otra mano se rellena con ceros y máscara 0.
- Si no se detectan manos, ambas manos quedan en cero (caso NONE típico).
- Pipeline de captura e inferencia usa la **misma extracción de features** para consistencia.
