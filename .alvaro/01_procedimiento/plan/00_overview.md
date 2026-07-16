# Plan de Procedimiento - Resumen General (00_overview.md)

Este documento presenta la visión general y la estructura maestro del estudio de ablación para la detección y clasificación de vehículos mediante cajas delimitadoras orientadas (OBB). El objetivo principal del estudio es evaluar el impacto de la limpieza de ruido de etiquetas basada en modelos generativos de inpainting (LaMa) y la aumentación de datos generativos armonizados (IC-Light) utilizando el modelo YOLO26s-OBB sobre el dataset SMART Challenge 2026 (MTC Perú).

---

## 1. Resumen Ejecutivo
El tráfico en el contexto urbano peruano representa un desafío que puede ser abordado mediante visión computacional. Sin embargo, el dataset del SMART Challenge 2026 presenta una limitación crítica: **solo se anotan vehículos en movimiento**, omitiendo sistemáticamente vehículos estacionados idénticos en apariencia. Esto introduce una señal contradictoria en el entrenamiento.

Para solucionar este sesgo, se propone un pipeline compuesto de tres fases:
1. **Fase 0 (Pseudo-labeling):** Detección exhaustiva de todos los vehículos y filtrado temporal compensando el movimiento del dron para aislar vehículos estacionados omitidos.
2. **Fase 1 (Limpieza con LaMa):** Remoción de vehículos estacionados y reconstrucción del fondo vial con Large Mask Inpainting (LaMa).
3. **Fase 2 (Aumentación con IC-Light):** Extracción de recortes reales de clases minoritarias, composición sobre los fondos limpios y armonización de iluminación con IC-Light.

El estudio de ablación evaluará de forma rigurosa el aporte individual de estas fases a través de 6 condiciones experimentales en un entorno de hardware mixto y bajo un cronograma ajustado con deadline el **22 de julio de 2026**.

---

## 2. Hipótesis
La integración secuencial de limpieza de ruido de etiquetas (LaMa) y aumentación espacial armonizada (IC-Light) incrementará el Macro AP-rIoU@[0.50:0.80] del modelo YOLO26s-OBB en un set de validación real e inalterado. La limpieza elimina la penalización por detectar autos estacionados no anotados en el entrenamiento, mientras que la aumentación mitiga el desbalance extremo de las clases minoritarias sin generar un *domain gap* destructivo.

---

## 3. Grafo de Dependencias entre Tareas

```mermaid
graph TD
    A[01_data_preparation.md] --> B[03_pseudo_labeling.md]
    A --> F1[06_training.md: Base 0, 1, 2]
    C[02_metric.md] --> H[07_evaluation.md]
    
    B --> D[04_lama_cleaning.md]
    D --> E[05_augmentation.md]
    D --> F2[06_training.md: Mejora A]
    
    E --> F3[06_training.md: Mejora B, C]
    D --> F3
    
    F1 --> H
    F2 --> H
    F3 --> H
```

### Flujos Paralelos:
- **Flujo A (Métrica):** `02_metric.md` puede implementarse y validarse de forma independiente en cualquier momento desde el Día 1.
- **Flujo B (Baselines):** `Base 0`, `Base 1` y `Base 2` de `06_training.md` pueden ejecutarse inmediatamente después de finalizar `01_data_preparation.md` sin esperar las fases de limpieza generativa.
- **Flujo C (Limpieza y Aumento en la Nube):** La limpieza con LaMa (`04_lama_cleaning.md`) y la extracción de recortes se ejecutan en Google Colab con GPU acelerada, mientras que los entrenamientos de ablación corren en paralelo en Kaggle.

---

## 4. Infraestructura de Cómputo

| Plataforma | GPU | VRAM | RAM | Almacenamiento | Rol Asignado |
|---|---|---|---|---|---|
| **Kaggle Notebooks (×5)** | P100 / 2×T4 | 16-32 GB | 29 GB | 20 GB (Persistente) + 73 GB (Scratch) | 1. **Fase 0:** Inferencia de pseudo-labeling.<br>2. **Fase 3 (Entrenamiento):** Entrenamientos paralelos de YOLO26s-OBB (las 5 condiciones de entrenamiento). Quota: 30h/semana por cuenta. |
| **Google Colab Free (×5)** | T4 | 15-16 GB | ~12 GB | ~78 GB (Efímero) | 1. **Fase 1 (Limpieza):** Inpainting con LaMa de forma masiva sobre la VM local.<br>2. **Fase 2 (Aumentación):** Inferencia con IC-Light para clases minoritarias.<br>3. **Evaluación:** Mapas Grad-CAM y benchmarks de ONNX/FP16. |
| **Google Drive Pro (×1)** | N/A | N/A | N/A | 1 TB (Nube) | Almacén central de datasets crudos, procesados en zip, parches extraídos y checkpoints de modelos. |

---

## 5. Logística de Datos

```mermaid
graph TD
    GD[Google Drive Pro 1TB] -->|Descarga directa rápida| GC[Google Colab VM]
    GC -->|Inferencia + LaMa + IC-Light| GC
    GC -->|Subida comprimida .zip| GD
    GD -->|Descarga de zip ligero| KN[Kaggle Notebooks]
    KN -->|kaggle datasets create| K[Kaggle Datasets]
    K -->|Attach a notebook| KN
```

1. El dataset crudo (`train.zip` de 40.3 GB) se descarga a una máquina virtual de **Google Colab** desde Google Drive Pro (aprovechando la velocidad interna de fibra de Google de >100 MB/s).
2. En Colab, se ejecuta el pipeline de la Fase 0 (detección de estacionados) y la Fase 1 (limpieza con LaMa). Al procesar, cada imagen se redimensiona a 640x640 para optimizar almacenamiento y ajustarse a la resolución de entrenamiento (`imgsz=640`).
3. El dataset limpio redimensionado (~4.3 GB) se comprime en un `.zip` y se sube de vuelta a Google Drive Pro en pocos minutos.
4. Para los entrenamientos en **Kaggle Notebooks**, se descarga este `.zip` ligero, se descomprime en el scratch disk y se registra como un Dataset Privado de Kaggle para que las 5 cuentas puedan montarlo de forma simultánea con latencia cero.

---

## 6. Cronograma de Ejecución (8 Días: 15–22 de Julio)

Las tareas críticas de la ruta principal están marcadas con una estrella (★).

| Día | Tarea en Google Colab (Interactivo/Generativo) | Tarea en Kaggle Notebooks (Entrenamientos/Filtros) | Entregable |
|---|---|---|---|
| **Día 1** (15 Jul) | ★ Descarga de `train.zip` a Colab.<br>★ Parseo de `train.csv` y conversión a YOLO OBB. | ★ Implementación y validación de la métrica `Macro AP-rIoU` con casos sintéticos. | `smart_dataset.yaml` y tests de la métrica aprobados. |
| **Día 2** (16 Jul) | ★ Inferencia con YOLO26s base y extracción de recortes de clases minoritarias. | ★ Lanzamiento de entrenamiento de **Base 1** (Data Cruda) en Kaggle.<br>★ Subida del dataset original a Kaggle Datasets. | `yolo26s_pseudo.pt` entrenado. Inicio de Base 1. |
| **Día 3** (17 Jul) | ★ Cálculo de homografía inter-frame (Ego-motion) en Colab.<br>★ Pseudo-labeling y auditoría manual (50 clips). | ★ Lanzamiento de entrenamiento de **Base 2** (Aumento Clásico) en Kaggle. | JSON con coordenadas de autos estacionados validadas. |
| **Día 4** (18 Jul) | ★ Ejecución de LaMa Inpainting en Colab GPU.<br>★ Redimensionamiento a 640x640 y subida del `.zip` limpio a Drive. | ★ Implementación del post-procesamiento del Filtro de Movimiento en Kaggle. | `dataset_lama_640.zip` subido a Drive. |
| **Día 5** (19 Jul) | ★ Armonización de imágenes compuestas con IC-Light en Colab GPU. | ★ Creación del dataset de Kaggle `smart-lama-cleaned`.<br>★ Lanzamiento de **Mejora A** (Data LaMa) en Kaggle. | Dataset sintético armonizado y Mejora A entrenando. |
| **Día 6** (20 Jul) | ★ Preparación final de los sets de datos consolidados. | ★ Lanzamiento de **Mejora B** (Cruda+Sintéticos) y **Mejora C** (Pipeline Completo). | Todos los modelos en entrenamiento en Kaggle. |
| **Día 7** (21 Jul) | ★ Extracción de mapas Grad-CAM de saliencia.<br>★ Benchmark de latencia de ONNX/FP16 en T4. | ★ Descarga de checkpoints y corrida de evaluación completa de las 6 condiciones (AP + Filtro). | Tabla completa de AP por clase y figuras de diagnóstico. |
| **Día 8** (22 Jul) | ★ **DEADLINE.** Compilación de resultados, redacción del borrador en LaTeX e integración. | N/A | Artículo listo para entrega. |

---

## 7. Índice de Documentos de Detalle

Para ver las especificaciones de diseño y flujos de trabajo de cada fase, consulte los siguientes archivos:

1. **[01_data_preparation.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/01_data_preparation.md):** Parseo, verificación estadística del dataset y split libre de fuga de datos.
2. **[02_metric.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/02_metric.md):** Especificación matemática y unit testing de Macro AP-rIoU.
3. **[03_pseudo_labeling.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/03_pseudo_labeling.md):** Detección de recall, ego-motion compensation y lógica de rastreo.
4. **[04_lama_cleaning.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/04_lama_cleaning.md):** Inpainting con LaMa y auditoría visual.
5. **[05_augmentation.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/05_augmentation.md):** Composición física realista y re-iluminación con IC-Light en Colab.
6. **[06_training.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/06_training.md):** Configuración de hiperparámetros de las 6 condiciones experimentales.
7. **[07_evaluation.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/07_evaluation.md):** Post-procesamiento temporal (filtro de movimiento) y diagnóstico de saliencia.
8. **[08_risks.md](file:///home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/.alvaro/01_procedimiento/plan/08_risks.md):** Mapa de contingencia y prioridades si el tiempo apremia.

---

## 8. Decisiones Técnicas Tomadas

| Decisión | Valor Seleccionado | Justificación Técnica |
|---|---|---|
| **Modelo Principal** | `YOLO26s-obb` | Aporta **+2.4 mAP** sobre la versión `nano`, mientras que pasar a la versión `medium` solo aporta **+0.5 mAP** a costa del doble de tiempo y recursos. |
| **Tamaño de Imagen** | `imgsz=640` | Permite entrenar con un batch size estable (batch=16 en Kaggle, batch=4 en GTX 1070 local) controlando el consumo de VRAM y previniendo errores OOM. |
| **Partición** | Aleatoria por Clip (`seed=42`) | Los frames de un mismo clip comparten el mismo fondo. El split a nivel de clip (80% train, 20% val) evita que frames del mismo clip terminen en ambos lados (*data leakage*). |
| **Dron Ego-motion** | Homografía ORB+RANSAC | Al grabarse desde un dron, el fondo se mueve sutilmente. El cálculo de la homografía permite compensar este movimiento antes de rastrear el centroide de los vehículos. |
| **Aumento Base 2** | `mosaic=1.0, mixup=0.15, copy_paste=0.3` | Hiperparámetros estándar e intermedios para aumentación clásica en datasets de teledetección/aéreos (DOTA). |
| **Aumento Mejora A/B/C** | Mínima (Sin Mosaic/MixUp/Copy-Paste) | Se desactivan las aumentaciones clásicas de YOLO para aislar y medir de forma pura el impacto de nuestras técnicas generativas. |
| **Ratio de Sintéticos** | Máx 50% por clase minoritaria | No superar 50% de la clase minoritaria y un límite de 1× de oversampling para buses articulados debido a la escasez de vehículos únicos (~5). |
| **Filtro de Movimiento** | Post-procesamiento idéntico en evaluación | Los modelos entrenados detectan vehículos estacionados. Se aplica un filtro de movimiento en inferencia para evaluar limpiamente contra la métrica de MTC. |
