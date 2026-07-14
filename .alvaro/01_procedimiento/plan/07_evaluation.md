# Evaluación y Diagnóstico (07_evaluation.md)

Este documento especifica el pipeline de evaluación cuantitativa y cualitativa para las 6 condiciones experimentales entrenadas en la Fase 06.

---

## 1. Objetivo
Evaluar rigurosamente el desempeño de cada modelo en un conjunto de validación compuesto al 100% de datos reales e inalterados, aplicando un filtro de movimiento temporal idéntico para aislar el error de detección y comparar el grado de generalización espacial.

---

## 2. El Pipeline de Inferencia y Post-Procesamiento

Los detectores entrenados identifican visualmente todos los vehículos en la imagen (parados y en movimiento). Dado que las anotaciones de validación (Ground Truth) solo contienen vehículos en movimiento, la evaluación directa de la inferencia sin procesar reportaría una alta tasa de falsos positivos en los autos estacionados. 

Para resolver esto, se especifica el siguiente pipeline secuencial de evaluación:

```
┌────────────────────────────────────────────────────────┐
│                  Inferencia Cruda                      │
│   Val images reales → YOLO26s-OBB (condición X)        │
│   → Detecciones crudas con umbral de confianza 0.001    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             Filtro de Movimiento Temporal              │
│   1. Tracking por proximidad y solapamiento.           │
│   2. Compensación de ego-motion del dron.               │
│   3. Clasificación de vehículos inmóviles.             │
│   → Descartar detecciones de vehículos estáticos.      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                Evaluación de AP                        │
│   Detecciones filtradas (solo vehículos en movimiento)   │
│   → Evaluador Macro AP-rIoU@[0.50:0.80]                │
└────────────────────────────────────────────────────────┘
```

El **Filtro de Movimiento** en inferencia se aplica de manera estrictamente **idéntica y constante** a las predicciones de las 6 condiciones de ablación para asegurar una comparación justa.

---

## 3. Especificación del Módulo: Filtro de Movimiento en Inferencia

* **Entrada:** Diccionario de detecciones crudas de todos los frames del clip de validación, matrices de homografía de ego-motion calculadas para el clip.
* **Proceso de Filtrado:**
  1. Agrupar las detecciones crudas por frame.
  2. Enlazar las predicciones a lo largo de los frames utilizando el algoritmo de tracking simple por distancia de centroide proyectado por homografía (especificado en la Fase 03).
  3. Para cada objeto rastreado, calcular su desplazamiento compensado acumulado:
     - Si el objeto se detecta en al menos $10$ frames y su desplazamiento neto compensado en píxeles es $< 8$ píxeles, se clasifica como **Vehículo Inmóvil**.
  4. **Acción de Filtrado:** Eliminar (borrar) todas las predicciones de las listas que correspondan a objetos clasificados como inmóviles.
  5. Retornar las predicciones filtradas (las cuales deben corresponder teóricamente solo a vehículos en movimiento).
* **Umbral de Confianza de Inferencia:** Se ejecuta la inferencia inicial con un umbral muy bajo de `conf = 0.001`. Esto permite retener las detecciones de baja confianza y construir la curva Precision-Recall completa necesaria para calcular el AP de forma exacta.

---

## 4. Tabla de Resultados Científicos a Reportar

Los resultados de la evaluación se tabularán en el artículo final de la siguiente forma:

| Condición | Macro AP-rIoU | AP@50 | AP@80 | Auto | Combi | Microbus | Minibus | Omnibus | Articulado | Camion | Mototaxi | Motocicleta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Base 0** (Zero-Shot) | | | | | | | | | | | | |
| **Base 1** (Data Cruda) | | | | | | | | | | | | |
| **Base 2** (Aumento Clásico) | | | | | | | | | | | | |
| **Mejora A** (Data LaMa) | | | | | | | | | | | | |
| **Mejora B** (Crudo + Sint.) | | | | | | | | | | | | |
| **Mejora C** (Pipeline Full) | | | | | | | | | | | | |

* **AP@50:** Promedio macro de AP al umbral mínimo de rIoU ($0.50$).
* **AP@80:** Promedio macro de AP al umbral más estricto de rIoU ($0.80$).
* **AP por Clase:** Promedio de AP a través de los 7 umbrales para cada una de las 9 clases vehiculares.

---

## 5. Diagnóstico Cualitativo: Mapas de Saliencia (Grad-CAM)

El objetivo de esta prueba es verificar empíricamente si la limpieza con LaMa elimina la correlación espacial del ruido del fondo, obligando al modelo a aprender semántica del objeto y no atajos del fondo.

* **Entrada:** Pesos entrenados de `Base 1` (Data Cruda) y `Mejora A` (Data LaMa), conjunto de $10$ imágenes de validación representativas que contengan tanto vehículos estacionados como en movimiento.
* **Proceso de Extracción:**
  1. Extraer los gradientes y activaciones de la última capa convolucional del backbone de YOLO26s para ambas condiciones de pesos.
  2. Generar los mapas de calor Grad-CAM superpuestos sobre las imágenes de validación.
* **Análisis Esperado (Justificación de la Hipótesis):**
  - **Base 1 (Modelo Ruidoso):** Debe mostrar activaciones concentradas en zonas del asfalto e intersecciones de forma estática (el modelo "espera" ver o no ver vehículos allí basado en el fondo).
  - **Mejora A (Modelo Limpio):** Debe mostrar focos de activación estrechamente localizados sobre los límites físicos y chasis de los vehículos en movimiento, ignorando las texturas limpias de la pista.

---

## 6. Benchmark de Exportación a Borde (Edge ONNX/FP16)

Para justificar la viabilidad en hardware de recursos limitados, se exporta el mejor modelo resultante.

* **Proceso de Exportación:**
  1. Exportar el archivo de pesos `best.pt` de la Mejor C a formato ONNX utilizando la utilidad nativa de Ultralytics:
     ```python
     model.export(format="onnx", half=True, dynamic=True)
     ```
  2. Medir el tiempo medio de ejecución y desviación estándar (latencia en milisegundos por frame) en una GPU Tesla T4 (Colab) para los siguientes batch sizes: $1$, $4$, $8$, y $16$.
  3. Reportar la latencia obtenida y los FPS promedio (Frames Per Second).
  4. Generar la curva Precisión vs Latencia para el artículo.

---

## 7. Criterios de Aceptación
- [ ] El filtro de movimiento debe procesar las detecciones crudas de validación sin fallos de indexación o desalineación de IDs.
- [ ] La tabla de resultados debe completarse en su totalidad para las 6 condiciones experimentales.
- [ ] Las imágenes Grad-CAM deben exportarse en formato de imagen de alta resolución para la diagramación del paper.
- [ ] El benchmark de velocidad debe reportarse con media y desviación estándar estadística.
