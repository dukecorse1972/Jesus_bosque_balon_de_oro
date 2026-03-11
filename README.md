# 🤟 Proyecto LSE: Reconocimiento de Gestos con TCN + MediaPipe Hands

Este proyecto permite la captura, entrenamiento y despliegue de un sistema de reconocimiento de Gestos Aislados de la **Lengua de Señas Española (LSE)**. Utiliza una arquitectura de **Redes Convolucionales Temporales (TCN)** alimentada por puntos de referencia (landmarks) de las manos extraídos mediante **MediaPipe**.

---

## 🚀 Características Principales

*   📸 **Captura Modular:** Herramienta para crear tu propio dataset con webcam.
*   🧠 **Arquitectura TCN:** Modelo robusto para secuencias temporales con conexiones residuales.
*   📊 **Validación Avanzada:** Métricas macro-F1 y división estratificada de datos.
*   📱 **Despliegue Flexible:** Exportación a TensorFlow Lite (con cuantización opcional).
*   ⏱️ **Inferencia en Tiempo Real:** Ventana deslizante para detección en vivo.

---

## 📂 Estructura del Proyecto

| Archivo | Descripción |
| :--- | :--- |
| `src/capture_dataset.py` | Captura de muestras con ventana temporal fija y remuestreo. |
| `src/dataset_utils.py` | Gestión de manifiestos, validación de clases y carga con `tf.data`. |
| `src/models.py` | Definición de la arquitectura TCN causal residual. |
| `src/train_tcn.py` | Script de entrenamiento completo y gestión de splits. |
| `src/eval.py` | Evaluación detallada sobre el conjunto de test. |
| `src/export_tflite.py` | Conversión a SavedModel y formato TFLite. |
| `src/infer_live.py` | Inferencia en vivo desde webcam (Keras o TFLite). |

---

## 🛠️ Guía de Uso

### 1. Instalación de Dependencias
Se recomienda el uso de un entorno virtual:

```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Captura del Dataset
Ejecuta la captura interactiva para grabar tus propios gestos:

```bash
python src/capture_dataset.py \
  --data_dir data \
  --window_seconds 1.5 \
  --target_fps 15 \
  --auto_period_seconds 3
```

**Controladores durante la captura:**
*   `1..9, 0, a..z`: Selecciona la clase de gesto activa.
*   `n`: Selecciona la clase `NONE` (fondo/sin gesto).
*   `ESPACIO`: Inicia la cuenta atrás y graba una secuencia.
*   `c` o `m`: Cambia entre modo automático y manual.
*   `r`: Reinicia el estado actual (repetir muestra).
*   `s`: Imprime estadísticas de la sesión.
*   `ESC` o `x`: Salir.

### 3. Entrenamiento del Modelo
Entrena la red TCN con el dataset generado:

```bash
python src/train_tcn.py \
  --data_dir data \
  --manifest manifest.jsonl \
  --gestures_yaml gestures.yaml \
  --epochs 40 \
  --batch_size 32 \
  --lr 1e-3 \
  --model_size small \
  --use_class_weights --augment
```

> [!NOTE]
> El dataset se divide automáticamente en **70% Entrenamiento**, **15% Validación** y **15% Test**, manteniendo la proporción de clases (estratificado).

### 4. Evaluación
Verifica el rendimiento del modelo en datos no vistos:

```bash
python src/eval.py \
  --data_dir data \
  --test_split outputs/splits/test.jsonl \
  --model_path outputs/checkpoints/best.keras \
  --gestures_yaml data/gestures.yaml \
  --save_cm_png
```

### 5. Exportación a TFLite
Convierte el modelo para su uso en dispositivos móviles o embebidos:

```bash
# Estándar
python src/export_tflite.py --model_path outputs/checkpoints/best.keras

# Con cuantización dinámica (más ligero)
python src/export_tflite.py --model_path outputs/checkpoints/best.keras --quantize_dynamic
```

### 6. Inferencia en Tiempo Real
Prueba el modelo directamente con tu webcam:

```bash
# Usando Keras (.keras)
python src/infer_live.py --model_path outputs/checkpoints/best.keras

# Usando TFLite (.tflite)
python src/infer_live.py --use_tflite --tflite_path outputs/tflite/model.tflite
```

---

## 🔍 Detalles Técnicos

### Representación de Datos
Cada muestra se guarda en formato `.npz` con una matriz `X` de forma `(T, F)`:
*   **T (Tiempo):** Frames en la ventana (Ej: 1.5s @ 15 FPS = 22 frames).
*   **F (Características):** 130 valores por frame.

**Desglose de características (F=130):**
*   Landmarks Mano Izquierda: 21 puntos × 3 (x, y, z) = **63**
*   Landmarks Mano Derecha: 21 puntos × 3 (x, y, z) = **63**
*   Máscaras de visibilidad: **2** (izq/der)
*   Lateralidad (Handedness): **2** (izq/der)

### Normalización
Para mejorar la robustez, los puntos de referencia se normalizan por mano:
1. Se traslada el origen a la muñeca (`wrist`).
2. Se escala la distancia entre la muñeca y el nudillo del dedo corazón (`middle_mcp`).

---

## 📁 Estructura de Salida (`outputs/`)

```text
outputs/
 ├─ checkpoints/     # Modelos guardados (.keras)
 ├─ splits/          # Registro de train/val/test (.jsonl)
 ├─ tflite/          # Modelos convertidos y etiquetas
 ├─ training_log.csv # Curvas de entrenamiento
 └─ config.json      # Hiperparámetros utilizados
```

---

## 📝 Notas
*   El número de clases se gestiona dinámicamente desde `data/gestures.yaml`.
*   La evaluación utiliza exactamente el mismo split de test generado durante el entrenamiento para garantizar resultados honestos.
