# Configuración de Entrenamientos y Ablación (06_training.md)

Este documento detalla las especificaciones de entrenamiento, la asignación de plataformas en la nube (Kaggle), y la justificación de los hiperparámetros para las 6 condiciones experimentales del estudio de ablación.

---

## 1. Objetivo
Entrenar el modelo YOLO26s-OBB bajo condiciones controladas e idénticas para aislar y cuantificar de manera exacta el aporte individual de la limpieza de ruido de etiquetas y de la aumentación de datos generativos.

---

## 2. Especificación de las 6 Condiciones Experimentales

Las 6 condiciones del estudio de ablación se configuran de la siguiente manera:

| Condición | Dataset de Entrenamiento | Configuración de Aumentación | Propósito Científico |
|---|---|---|---|
| **Base 0 (Zero-Shot)** | Ninguno (Sin entrenamiento) | N/A | Evaluar el desempeño base de YOLO26s-OBB pre-entrenado en datasets generales (COCO/DOTAv2) en la flota y vistas del Perú. AP esperado: muy bajo. |
| **Base 1 (Data Cruda)** | Original `train.csv` (~43,400 frames) | **Mínima:** Sin aumentaciones artificiales de YOLO (Mosaic, MixUp, Copy-Paste inactivos). | Línea base real del dataset original con ruido de etiquetas de autos estacionados y desbalance extremo. |
| **Base 2 (Aumento Clásico)** | Original `train.csv` (~43,400 frames) | **Full YOLO:** Aumentaciones estándar activadas (Mosaic, MixUp, Copy-Paste, Erasing). | Cuantificar el rendimiento máximo alcanzable con técnicas estándar de augmentación de YOLOv8/11/26. |
| **Mejora A (Data LaMa)** | Dataset limpiado con LaMa (~43,400 frames) | **Mínima:** Sin aumentaciones artificiales. | Medir de forma aislada el impacto de **limpiar** las etiquetas contradictorias (borrar autos estacionados) de la base de entrenamiento. |
| **Mejora B (Aumento Generativo Crudo)** | Original `train.csv` + Sintéticos IC-Light (~43,800 frames) | **Mínima:** Sin aumentaciones artificiales. | Evaluar si la inyección de clases minoritarias generadas artificialmente es efectiva si los datos base siguen teniendo ruido. |
| **Mejora C (Pipeline Completo)** | Dataset LaMa + Sintéticos IC-Light (~43,800 frames) | **Mínima:** Sin aumentaciones artificiales. | Configuración propuesta del estado del arte. Mide el impacto combinado de limpieza de datos base + aumentación armonizada. |

---

## 3. Especificación del Modelo Común e Hiperparámetros

Todos los entrenamientos (Base 1, Base 2, Mejoras A, B, C) compartirán exactamente los mismos hiperparámetros de optimización para asegurar la validez de la comparación:

```yaml
# smart_training_hyperparameters.yaml
model: yolo26s-obb.pt        # Modelo base con pesos de COCO/DOTA
imgsz: 640                   # Resolución controlada para VRAM
epochs: 100                  # 100 épocas de entrenamiento completo
patience: 20                 # Detener si mAP50-95 no mejora en 20 épocas
batch: 16                    # Batch size para P100 en Kaggle
optimizer: AdamW
lr0: 0.001
lrf: 0.01                    # Cosine learning rate decay
weight_decay: 0.0005
seed: 42                     # Semilla determinista fija
device: 0                    # Index de GPU

# Pérdidas por defecto de Ultralytics YOLO OBB
box: 7.5                     # Peso de la pérdida de caja delimitadora
cls: 0.5                     # Peso de la pérdida de clasificación
dfl: 1.5                     # Peso de la pérdida de distribución focal
```

---

## 4. Control de Aumentaciones por Condición

Las transformaciones geométricas básicas (que no crean contenido nuevo sino que rotan, escalan o trasladan las imágenes existentes) son necesarias para que el modelo OBB aprenda a detectar objetos en ángulos arbitrarios desde el dron. Se aplican a todas las condiciones de la siguiente manera:

```yaml
# Aumentaciones Geométricas Comunes (Válidas para TODAS las condiciones)
fliplr: 0.5                  # Volteo horizontal al 50%
flipud: 0.5                  # Volteo vertical al 50%
degrees: 180.0               # Rotación libre de 0 a 180 grados
scale: 0.5                   # Escalamiento de 0.5x a 1.5x
translate: 0.1               # Traslación de +-10% en el lienzo
```

### 4.1 Configuración de Aumentaciones Específicas

#### Base 1 y Mejoras A, B, C (Mínima Aumentación)
Se desactivan los procesos que combinan o pegan objetos artificiales de forma clásica:
* `mosaic: 0.0`
* `mixup: 0.0`
* `copy_paste: 0.0`
* `erasing: 0.0`
* `hsv_h: 0.015`, `hsv_s: 0.3`, `hsv_v: 0.2` (Color conservador)

#### Base 2 (Aumentación Clásica de YOLO)
Se activan los parámetros estándar e intermedios del framework de Ultralytics para detección OBB aérea:
* `mosaic: 1.0` (Combinación de 4 imágenes activada)
* `mixup: 0.15` (Mezcla lineal de dos imágenes activada al 15% de probabilidad)
* `copy_paste: 0.3` (Pegado crudo de vehículos de otras imágenes activado al 30% de probabilidad)
* `erasing: 0.4` (Borrado aleatorio de parches para simular oclusión al 40% de probabilidad)
* `close_mosaic: 10` (Desactivar Mosaic durante las últimas 10 épocas para estabilización final)
* `hsv_h: 0.015`, `hsv_s: 0.7`, `hsv_v: 0.4` (Color por defecto de YOLO)

---

## 5. Logística y Paralelización de Cómputo (Kaggle)

Los entrenamientos se paralelizan en las 5 cuentas de Kaggle de los integrantes del equipo para finalizar dentro del cronograma. Cada entrenamiento requiere aproximadamente entre $10$ y $14$ horas en una GPU Tesla P100 (16 GB).

### 5.1 Distribución de Entrenamientos por Cuenta

| Cuenta Kaggle | Tarea Asignada | Dataset Requerido | Épocas |
|---|---|---|---|
| **Cuenta #1** | Entrenamiento: **Base 1** | `smart-original` (Original YOLO format) | 100 |
| **Cuenta #2** | Entrenamiento: **Base 2** | `smart-original` | 100 |
| **Cuenta #3** | Entrenamiento: **Mejora A** | `smart-lama-cleaned` (Cleaned images) | 100 |
| **Cuenta #4** | Entrenamiento: **Mejora B** | `smart-original` + `smart-synthetic` | 100 |
| **Cuenta #5** | Entrenamiento: **Mejora C** | `smart-lama-cleaned` + `smart-synthetic` | 100 |

### 5.2 Gestión de Checkpoints y Almacenamiento
* El entrenamiento de Ultralytics genera los checkpoints en la ruta `/kaggle/working/runs/obb/`.
* Se debe configurar el guardado de checkpoints intermedios cada 10 épocas para prevenir pérdidas por desconexión del notebook (máximo 9 horas por sesión en Kaggle).
* Al finalizar el entrenamiento, descargar inmediatamente los pesos finales `best.pt` y `last.pt` y subirlos a la cuenta central de **Google Drive Pro** con nomenclatura clara (ej. `best_base1.pt`, `best_mejora_a.pt`).

---

## 6. Criterios de Aceptación
- [ ] Los 5 entrenamientos en la nube deben completarse sin excepciones de desbordamiento de memoria (OOM).
- [ ] Los logs de entrenamiento (`results.csv` o Tensorboard logs) deben descargarse para graficar curvas de pérdida.
- [ ] La validación interna al final de cada entrenamiento en Kaggle debe reportarse sobre el val split (el cual siempre es de datos reales e inalterados).
- [ ] Los archivos de pesos `.pt` descargados deben pesar exactamente lo mismo (~20 MB para YOLO26s), garantizando su integridad.
