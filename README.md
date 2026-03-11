# Proyecto LSE (gestos aislados) con TCN + MediaPipe Hands

Proyecto en Python para capturar dataset propio, entrenar un modelo TCN en TensorFlow/Keras, evaluarlo, exportarlo a TFLite e inferir en vivo desde webcam usando solo manos.

## Estructura

- `src/capture_dataset.py`: captura de muestras por ventana temporal fija y remuestreo a `target_fps`.
- `src/dataset_utils.py`: lectura de manifest, validación de clases, split estratificado, `tf.data`, augment y class weights.
- `src/models.py`: arquitectura TCN causal residual.
- `src/metrics.py`: callback para macro-F1 en validación.
- `src/train_tcn.py`: entrenamiento completo y guardado explícito de splits.
- `src/eval.py`: evaluación sobre el split de test guardado.
- `src/export_tflite.py`: export a SavedModel + TFLite con validación de clases y labels.
- `src/infer_live.py`: inferencia en vivo Keras o TFLite con ventana deslizante.
- `data/gestures.yaml`: mapeo editable `id_to_name` y `name_to_id`.
- `data/manifest.jsonl`: registro de muestras guardadas (paths relativos a `data/`).

## 1) Instalar dependencias

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
2) Capturar dataset (webcam)
python src/capture_dataset.py --data_dir data --window_seconds 1.5 --target_fps 15 --auto_period_seconds 3
Teclas de captura

1..9, 0, a..z: seleccionan clases según el mapa mostrado en consola.

n: selecciona NONE.

SPACE: inicia cuenta atrás y graba una secuencia.

c o m: activa/desactiva modo automático.

r: repetir (reinicia estado sin guardar).

s: imprime contadores de sesión por gesto.

x o ESC: salir.

Formato guardado por muestra (.npz)

X: float32, shape (T, F).

T = round(window_seconds * target_fps).

Con 1.5 s y 15 FPS, T = 22.

F = 130.

Features por frame (F=130)

LEFT: 21*(x,y,z)=63 normalizados.

RIGHT: 63 normalizados.

mask_left, mask_right (2).

handedness_left, handedness_right (2).

Normalización por mano:

Restar wrist (lm0).

Escalar por distancia wrist(0)->middle_mcp(9).

Sin rotación adicional.

3) Entrenar
python src/train_tcn.py \
  --data_dir data \
  --manifest manifest.jsonl \
  --gestures_yaml gestures.yaml \
  --epochs 40 \
  --batch_size 32 \
  --lr 1e-3 \
  --seed 42 \
  --use_class_weights \
  --augment \
  --model_size small

Split por defecto: 70/15/15 estratificado.

Salida principal:

outputs/checkpoints/best.keras

outputs/checkpoints/last.keras

outputs/training_log.csv

outputs/config.json

outputs/splits/train.jsonl

outputs/splits/val.jsonl

outputs/splits/test.jsonl

outputs/labels.txt

4) Evaluar
python src/eval.py \
  --data_dir data \
  --test_split outputs/splits/test.jsonl \
  --model_path outputs/checkpoints/best.keras \
  --gestures_yaml data/gestures.yaml \
  --save_cm_png

Imprime:

accuracy

macro-F1

F1 para clase NONE

matriz de confusión

5) Exportar a TFLite
python src/export_tflite.py \
  --model_path outputs/checkpoints/best.keras \
  --gestures_yaml data/gestures.yaml \
  --verify

Con quantización dinámica opcional:

python src/export_tflite.py \
  --model_path outputs/checkpoints/best.keras \
  --gestures_yaml data/gestures.yaml \
  --quantize_dynamic \
  --verify

Salidas:

outputs/saved_model/

outputs/tflite/model.tflite

outputs/tflite/labels.txt

6) Inferencia en vivo
Keras
python src/infer_live.py \
  --model_path outputs/checkpoints/best.keras \
  --gestures_yaml data/gestures.yaml \
  --target_fps 15 \
  --stride_frames 1
TFLite
python src/infer_live.py \
  --use_tflite \
  --tflite_path outputs/tflite/model.tflite \
  --gestures_yaml data/gestures.yaml \
  --target_fps 15 \
  --stride_frames 1

Flags útiles:

--threshold 0.5: si la confianza top-1 < threshold, fuerza NONE.

--show_top3: muestra top-3 en overlay.

--smooth_n 1: promedio móvil de predicciones.

--stride_frames 1: frecuencia de predicción sobre la ventana deslizante.

Notas importantes

El número de clases se toma de gestures.yaml, no de un entero hardcodeado.

El entrenamiento guarda los splits usados; la evaluación debe reutilizarlos.

El manifest.jsonl puede contener rutas con \ o /; el loader las normaliza.

Captura e inferencia usan la misma extracción de features, y la captura remuestrea la ventana temporal a target_fps antes de guardar