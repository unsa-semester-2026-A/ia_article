# Rotated Object Detection via Scale-Invariant Mahalanobis Distance in Aerial Images

- **Key**: Wen2022MahalanobisRotatedDetection
- **Year**: 2022
- **Venue**: IEEE Geoscience and Remote Sensing Letters (LGRS)

## Resumen
Este artículo propone una nueva función de pérdida llamada Pérdida de Distancia de Mahalanobis (MDL) para la detección de objetos orientados en imágenes aéreas utilizando una representación de ocho parámetros (las coordenadas de los cuatro vértices del rectángulo). Los autores demuestran que las pérdidas comunes basadas en normas $\ell_n$ (L1, L2, smooth L1) dependen de la métrica de Minkowski, la cual no es invariante a la escala. Como resultado, las fluctuaciones de escala en las imágenes aéreas causan inestabilidad en el entrenamiento e inconsistencia con la métrica de evaluación Rotational IoU (SkewIoU). Al modelar los cuatro vértices de una caja delimitadora orientada (OBB) como una distribución bidimensional y calcular su matriz de covarianza, MDL utiliza la distancia de Mahalanobis para medir la discrepancia entre las predicciones y el ground truth. Debido a la propiedad de invarianza de escala de esta métrica, MDL proporciona curvas de pérdida más estables y correlacionadas con la métrica SkewIoU. Para corregir la discontinuidad en las fronteras debido a la periodicidad angular, se evalúan las cuatro permutaciones posibles de los vértices y se selecciona el valor de pérdida mínimo. Implementado sobre una arquitectura CenterNet (sin anclas), MDL-p (usando la covarianza de la caja predicha) obtiene un mAP del 76.16% en DOTA-v1.0, superando significativamente a las pérdidas smooth L1 tradicionales.

## Secciones y Subsecciones

### I. Introducción
Describe la importancia de la orientación en la detección de objetos aéreos debido a la alta densidad y colocación arbitraria de las instancias. Compara las representaciones OBB de cinco parámetros $(x, y, w, h, \theta)$ con las de ocho parámetros (coordenadas de vértices).
* **Problemas atacados**: Inconsistencia y fluctuaciones drásticas en la pérdida de regresión provocadas por cambios de escala en cajas orientadas al usar normas $\ell_n$ basadas en Minkowski.
* **Limitaciones de ese entonces**: En métodos de 5 parámetros, pequeñas desviaciones en el ángulo causan caídas masivas no lineales en SkewIoU. En métodos de 8 parámetros, las pérdidas L1/L2 clásicas no son invariantes a la escala (duplicar el tamaño de la caja cuadruplica la pérdida smooth L1 aun manteniendo idéntico el ratio de error de localización, ver Figura 1).
* **Soluciones alcanzadas**: Uso de la distancia de Mahalanobis para formular la pérdida de regresión de vértices (MDL), logrando curvas de coste insensibles al tamaño absoluto de las cajas y correlacionadas con SkewIoU.

### II. Enfoque Propuesto
Detalla el diseño matemático de la Pérdida de Distancia de Mahalanobis (MDL) y su integración en un detector CenterNet modificado.

* **Problemas atacados**: Modelado estadístico de una caja rectangular como distribución espacial y corrección de la discontinuidad del ángulo periódico en el extremo de la rotación.
* **Limitaciones de ese entonces**: Calcular el SkewIoU directo durante el entrenamiento es inviable debido a que no es diferenciable y su cálculo es extremadamente complejo. Las formulaciones previas de 8 parámetros sufren de saltos abruptos de gradiente en los límites de los ejes debido al orden de los vértices.
* **Soluciones alcanzadas**: Modelado de covarianza bidimensional e implementación de una permutación circular de 4 posiciones para estabilizar la pérdida en las fronteras.

#### II.A Pérdida de Distancia de Mahalanobis para Cajas Delimitadoras Orientadas
Presenta la ecuación de la distancia de Mahalanobis y el cálculo de la matriz de covarianza $\Sigma$ de tamaño $2 \times 2$ usando los cuatro vértices de la caja (Ecuación 2).
* **Problemas atacados**: Representación matricial de las correlaciones espaciales de los vértices.
* **Limitaciones de ese entonces**: Las distancias euclidianas tratan cada dimensión y vértice por separado, ignorando cómo están correlacionadas las coordenadas de los vértices entre sí en un plano rotado.
* **Soluciones alcanzadas**: Cálculo de $\Sigma$ a partir de las coordenadas $(x, y)$ de los cuatro vértices. La covarianza captura la orientación y elongación inherentes del rectángulo de forma independiente a cuál sea el vértice de inicio, resultando en una métrica MDL unificada (Ecuación 3).

#### II.B Análisis de la Pérdida de Distancia de Mahalanobis
Compara cualitativa y cuantitativamente las curvas de MDL frente a smooth L1 ante variaciones aisladas de ángulo, escala, desplazamiento y relación de aspecto (Figuras 3 y 4).
* **Problemas atacados**: Demostrar la consistencia métrica frente a SkewIoU.
* **Limitaciones de ese entonces**: Las pérdidas smooth L1 muestran pendientes excesivamente pronunciadas e inestables que desvían el aprendizaje en cajas grandes o muy alargadas.
* **Soluciones alcanzadas**: Demostración matemática de que la curva de MDL permanece perfectamente plana e invariable cuando solo cambia el factor de escala, imitando exactamente el comportamiento de $1-\text{SkewIoU}$.

#### II.C Diseño de la Función de Pérdida General
Describe la arquitectura CenterNet modificada (ResNet-101 y cuello en U) que genera tres mapas de salida: Heatmap $P$, Vértices $B$ (cuatro vectores al centro) y Offset $O$.
* **Problemas atacados**: Optimización simultánea de la detección de centros de cajas, vectores de vértices y errores de cuantización por submuestreo.
* **Limitaciones de ese entonces**: CenterNet horizontal clásico usa pérdidas L1 simples para cajas axiales y no modela la rotación.
* **Soluciones alcanzadas**: 1) Uso de Focal Loss modificada para el Heatmap de centros. 2) Aplicación de MDL con suavizado de frontera mediante búsqueda de la pérdida mínima entre 4 rotaciones de índices (Ecuación 4). 3) Adición de una pérdida de offset compensatoria formulada también mediante distancia de Mahalanobis adaptada al tamaño del OBB (Ecuación 5).

### III. Experimentos
Presenta los resultados empíricos obtenidos sobre el benchmark DOTA-v1.0.

* **Problemas atacados**: Validación de MDL frente a smooth L1 y otros detectores orientados de la literatura.
* **Limitaciones de ese entonces**: Falta de comparaciones con idénticas arquitecturas base para aislar el efecto de la pérdida de regresión.
* **Soluciones alcanzadas**: Experimentos rigurosos entrenando CenterNet durante 50 épocas con optimizador Adam.

#### III.A Conjunto de Datos
Describe el preprocesamiento de DOTA-v1.0.
* **Problemas atacados**: Adaptar las imágenes de DOTA a la red CenterNet.
* **Limitaciones de ese entonces**: Imágenes satelitales gigantescas no pueden procesarse directamente en CenterNet.
* **Soluciones alcanzadas**: Troceado en parches de 600x600 píxeles con solape de 100 en escalas de 0.5 y 1.0, generando un conjunto de entrenamiento de 69,337 imágenes y 35,777 de prueba.

#### III.B Detalles de Implementación y Prueba
Detalla el flujo de inferencia y la combinación de predicciones.
* **Problemas atacados**: Reconstrucción final de las cajas y supresión de duplicados.
* **Limitaciones de ese entonces**: El submuestreo por un factor de 4 introduce imprecisiones de píxel al reconstruir la escala original.
* **Soluciones alcanzadas**: Selección de las 500 mejores propuestas de centros, adición del offset ajustado por Mahalanobis, escalado inverso por factor 4 y aplicación de NMS orientado con umbral de 0.1.

#### III.C Estudio de Ablación
Analiza las variantes de MDL y la comparación contra smooth L1.
* **Problemas atacados**: Determinar el origen óptimo para calcular la matriz de covarianza $\Sigma$ y verificar la ganancia neta de MDL.
* **Limitaciones de ese entonces**: Especular si la covarianza calculada con los puntos del ground truth (fija) es mejor que con las predicciones (dinámica).
* **Soluciones alcanzadas**: 1) Calcular $\Sigma$ con los vectores predichos (MDL-p) alcanza un mAP del 76.16% frente al 74.33% usando el ground truth (MDL-t), sugiriendo que MDL-p facilita los gradientes de regresión. 2) MDL-p supera a smooth L1 por un margen neto de 2.18% mAP con la misma arquitectura base, demostrando curvas de entrenamiento mucho más estables (ver Figura 6).

#### III.D Comparaciones Adicionales
Compara MDL-p frente a detectores de cinco y ocho parámetros del estado del arte (Tabla II).
* **Problemas atacados**: Validar MDL frente a otros solucionadores del problema de discontinuidad de frontera.
* **Limitaciones de ese entonces**: Métodos de 5 parámetros pierden mAP por desalineación angular. Métodos de 8 parámetros (ej. RSDet, BBAVectors) sufren al usar smooth L1 no escalable.
* **Soluciones alcanzadas**: MDL-p alcanza el mAP más alto (76.16%), superando a RoI Transformer (69.56%), SCRDet (72.61%), RSDet (74.1%) y BBAVectors (75.36%). Obtiene los mejores rendimientos en clases complejas y pequeñas como Puertos, Vehículos Grandes y Helicópteros.

### IV. Conclusión
* **Problemas atacados**: Resumen de aportes y limitaciones persistentes.
* **Limitaciones de ese entonces**: El problema de la periodicidad angular y discontinuidad en los bordes sigue requiriendo trucos heurísticos de ordenamiento y permutación de pérdidas.
* **Soluciones alcanzadas**: Validación de la invarianza de escala de Mahalanobis para corregir la inestabilidad de Minkowski. Se propone como trabajo futuro investigar soluciones fundamentales para la discontinuidad angular y extender MDL a la detección de polígonos genéricos de más de 4 lados.
