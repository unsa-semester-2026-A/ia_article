# Mapa de Riesgos y Contingencias (08_risks.md)

Este documento detalla la matriz de riesgos del proyecto, las acciones preventivas (mitigaciones), las acciones correctivas (contingencias) y los criterios de priorización en caso de limitaciones severas de tiempo o recursos antes del deadline del **22 de julio de 2026**.

---

## 1. Matriz de Riesgos

| ID | Evento de Riesgo | Probabilidad | Impacto | Acción Preventiva (Mitigación) | Acción Correctiva (Contingencia si falla) |
|---|---|---|---|---|---|
| **R1** | **El pseudo-labeler temporal tiene baja precisión (<90%)** | Media | Alto | Ajustar finamente los umbrales de detección y la cantidad de frames mínimos en el set de validación de prueba en Colab antes de correr sobre todo el dataset. | Ajustar el filtro con una revisión manual más estricta o reducir el pipeline de inpainting solo a los 200 clips de entrenamiento con mayor tasa de error visible. |
| **R2** | **LaMa genera artefactos visuales notorios tras el inpainting** | Baja | Medio | Realizar una dilatación de máscara conservadora (10 px) y auditar visualmente una muestra de 100 imágenes. | **Fallback de Máscara Simple (Nautilus):** Si LaMa falla, sustituir la región del auto estacionado por un cuadro negro o ruido gaussiano. Esto sirve como baseline de control y valida si la reconstrucción fotorrealista es indispensable. |
| **R3** | **IC-Light causa OOM (Out of Memory) o es lento** | Alta | Medio | Diseñar el pipeline para procesar en Colab T4 con resoluciones controladas de 512×512 píxeles. | Limitar la resolución a $512\times512$ píxeles y procesar en lotes a través de notebooks de Colab sincronizados con Google Drive Pro. |
| **R4** | **El oversampling generativo causa sobreajuste (memorización de instancia)** | Media | Alto | Aplicar transformaciones visuales agresivas (color jitter, escala, ruido y rotaciones adicionales) a cada recorte antes de componerlos. | Reportar de forma transparente la brecha de AP entre entrenamiento y validación por clase, y documentar la limitación en el artículo científico (para articulado ~5 buses únicos). |
| **R5** | **La cuota de GPU en Kaggle (30h/cuenta) se agota antes de finalizar** | Media | Alto | Distribuir el entrenamiento de las 5 condiciones en las 5 cuentas individuales de los integrantes (1 cuenta por condición, consumiendo ~12h cada una). | Utilizar Colab Free (GPU T4) con entrenamientos resumidos (congelando las primeras 15 capas del backbone de YOLO y reduciendo a 50 épocas) o ejecutar entrenamientos en segundo plano usando la opción 'Save Version -> Run All in Background' de Kaggle. |
| **R6** | **El Filtro de Movimiento en evaluación es muy estricto o muy flexible** | Media | Alto | Calibrar los parámetros del filtro utilizando el subconjunto de entrenamiento donde se conocen los vehículos estacionados reales. | Auditar visualmente el recall de las predicciones filtradas en 20 clips de validación y ajustar el umbral de píxeles (`motion_threshold`) en pasos de $\pm2$ píxeles. |
| **R7** | **No se llega a redactar todo el artículo antes del 22 de julio** | Media | Alto | Iniciar la redacción de las secciones de Introducción, Metodología y Related Work desde el Día 2 en paralelo con los entrenamientos. | Implementar el principio de **Ablación Mínima Viable** (ver Sección 2) para reducir el volumen de experimentos requeridos si el tiempo se agota. |

---

## 2. Priorización de Experimentos (Ablación Mínima Viable)

Si surgen fallas técnicas o el tiempo al 22 de julio es muy corto, se aplicará el siguiente plan de reducción de alcance de forma secuencial:

### Nivel 1: Alcance Completo (6 Condiciones)
* **Condiciones:** Base 0, Base 1, Base 2, Mejora A, Mejora B, Mejora C.
* **Requisito:** 5 cuentas de Kaggle activas y pipeline de IC-Light completado.

### Nivel 2: Reducción Moderada (5 Condiciones)
* **Acción:** Eliminar **Base 0 (Zero-Shot)**.
* **Justificación:** Aunque demuestra la necesidad de fine-tuning, el aporte científico de Base 0 es el menos relevante y más predecible. Ahorra tiempo de inferencia y formateo.

### Nivel 3: Reducción Crítica (4 Condiciones)
* **Acción:** Eliminar **Mejora B (Aumento Generativo Crudo)**.
* **Justificación:** Conserva los baselines (Base 1 y Base 2) y el pipeline de limpieza (Mejora A) y completo (Mejora C). Permite responder a la efectividad de limpiar y aumentar, aunque se pierde el análisis de si la aumentación sola sin limpieza es dañina.

### Nivel 4: Ablación Mínima Viable (3 Condiciones)
* **Acción:** Entrenar únicamente **Base 1 (Data Cruda)**, **Mejora A (Data LaMa)** y **Mejora C (Pipeline Completo)**.
* **Justificación:** Este es el núcleo indispensable de la tesis. Demuestra la línea base ruidosa, el impacto puro de remover el ruido (LaMa) y el impacto final al balancear las clases (IC-Light). Se pueden entrenar las 3 condiciones en una sola noche usando 3 cuentas de Kaggle.

---

## 3. Plan de Contingencia del Pipeline de Datos (Fallback de Copia Directa)

Si la cuota de espacio de Google Drive Pro falla o el tiempo de descarga/subida de los datasets en Colab es excesivo:

1. **Fallback de Descarga Directa de Kaggle:** Descargar el dataset original directamente dentro de cada notebook de Kaggle usando la API nativa de la competencia (velocidad de red interna de Google a >100 MB/s).
2. **Fallback de Procesamiento en Kaggle (Inpainting Nube):**
   - Ejecutar el inpainting de LaMa directamente dentro de un notebook de Kaggle utilizando la GPU de la cuenta.
   - Guardar el resultado en `/kaggle/working/` y exportarlo como un nuevo dataset privado sin pasar por almacenamiento local.
   - Esto elimina la necesidad de subir o descargar archivos de 40 GB en equipos locales.
