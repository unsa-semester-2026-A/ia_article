# Rotated Object Detection via Scale-invariant Mahalanobis Distance in Aerial Images

- **Key**: Wen2022MahalanobisRotatedDetection
- **Year**: 2022
- **Venue**: IEEE Geoscience and Remote Sensing Letters (LGRS)

## Resumen
Este artículo presenta una función de pérdida novedosa llamada Pérdida de Distancia de Mahalanobis (MDL, Mahalanobis Distance Loss) para abordar los desafíos de la regresión de cajas orientadas (OBB) de ocho parámetros en imágenes aéreas. En la detección de objetos aéreos, las tomas contienen objetos en múltiples escalas espaciales y con orientaciones libres. Las funciones de pérdida tradicionales como las normas L_n (L1, L2 y L1 suave) se basan en la distancia no invariante de Minkowski, lo que provoca que el valor de pérdida fluctúe fuertemente cuando cambia la escala de los objetos. Esto genera inconsistencia con la métrica de evaluación real (SkewIoU rotado) e inestabilidad durante el entrenamiento. Al modelar la caja delimitadora como una distribución bidimensional discreta formada por sus cuatro vértices, MDL aprovecha la invarianza de escala de la distancia de Mahalanobis. Esto resulta en gradientes estables e insensibles al tamaño de la instancia, mejorando la coincidencia con SkewIoU. Para corregir la discontinuidad en los límites angulares, los autores aplican una técnica de pérdida mínima permutada. Implementada sobre un detector libre de anclajes (CenterNet), MDL-p logra un 76.16% de mAP en el dataset DOTA-v1.0, superando de forma marcada a los esquemas clásicos basados en pérdidas L1 suaves.

## Secciones y Subsecciones

### I. Introduction
Se expone el auge de la detección de objetos orientados en imágenes de percepción remota y la transición de las representaciones tradicionales horizontales a las orientadas. Se explica por qué las métricas de pérdida basadas en la distancia de Minkowski son perjudiciales en este dominio.
* **Problemas atacados**: La inconsistencia entre la optimización de la red (basada en pérdidas Minkowski de tipo L_n) y la métrica de validación final (rotational IoU o SkewIoU), y la inestabilidad en el entrenamiento cuando se presentan escalas sumamente variadas.
* **Limitaciones de ese entonces**: A diferencia del caso horizontal, la métrica SkewIoU en OBB es altamente no lineal, no diferenciable y compleja de optimizar directamente en pérdidas de tipo IoU. Como solución, los métodos de cinco u ocho parámetros dependían de pérdidas Minkowski como Smooth L1, que no son invariantes a la escala (Fig. 1) y penalizan desproporcionadamente a los objetos grandes frente a los pequeños.
* **Soluciones alcanzadas**: Introducción de la distancia de Mahalanobis como métrica de pérdida matemática de regresión de cajas orientadas, la cual posee invarianza de escala nativa al considerar la covarianza espacial de los vértices del cuadrilátero.

### II. Proposed Approach
Descripción detallada de la formulación matemática de la Pérdida de Distancia de Mahalanobis (MDL) aplicada a cajas orientadas y la arquitectura base basada en CenterNet.
* **Problemas atacados**: Modelar una representación métrica invariante a la escala y la rotación para un conjunto ordenado de 4 esquinas de caja orientada.
* **Limitaciones de ese entonces**: Las pérdidas normales no consideran la correlación espacial entre las variables de coordenadas $x$ e $y$ de los vértices de la caja.
* **Soluciones alcanzadas**: Representación matemática de la caja OBB como una distribución de probabilidad discreta de 2D y el cálculo de la matriz de covarianza asociada para computar la distancia de Mahalanobis.

#### A. Mahalanobis Distance Loss for Oriented Bounding Box
Formulación matemática de la pérdida MDL (Eq. 1, 2 y 3).
* **Problemas atacados**: Definir la distancia de Mahalanobis sobre las coordenadas de vértices de la caja delimitadora.
* **Limitaciones de ese entonces**: La distancia de Mahalanobis requiere una matriz de covarianza $\Sigma$ que típicamente asume conjuntos de datos continuos y densos, no cuadriláteros simples de 4 puntos.
* **Soluciones alcanzadas**: Representación de los 4 vértices del cuadrilátero como vectores bidimensionales para calcular una covarianza espacial $\Sigma$ de $2 \times 2$. Esto permite promediar la distancia de Mahalanobis de las 4 esquinas con respecto al ground truth, resultando en un entrenamiento numéricamente estable.

#### B. Analysis of Mahalanobis Distance Loss
Comparativa analítica de las curvas de pérdida de MDL frente a Minkowski bajo cambios aislados de escala, ángulo, desplazamiento del centro y relación de aspecto.
* **Problemas atacados**: Validación matemática del comportamiento de la pérdida MDL bajo distorsiones geométricas aisladas.
* **Limitaciones de ese entonces**: Las pérdidas Minkowski tradicionales (L1 y Smooth L1) se vuelven más empinadas de forma no lineal conforme aumenta la escala del objeto, distorsionando los gradientes de retropropagación.
* **Soluciones alcanzadas**: Demostración mediante gráficas (Fig. 3 y 4) de que las curvas de MDL son invariantes a cambios de escala física y siguen fielmente la tendencia de la métrica física 1-SkewIoU en todas las variaciones de rotación, centrado y forma.

#### C. Overall Loss Function Design
Integración de MDL dentro del detector libre de anclajes CenterNet, detallando las ramas de predicción: Heatmap, Box Vertices y Offset.
* **Problemas atacados**: Detección de objetos sin anclajes resolviendo discontinuidades en las fronteras angulares y el error por discretización espacial del mapa de características.
* **Limitaciones de ese entonces**: Al igual que todos los detectores de 8 parámetros, las regresiones de esquinas sufren discontinuidades abruptas en los bordes del rango angular por la periodicidad física de los ángulos trigonométricos.
* **Soluciones alcanzadas**: 1) Rama Heatmap: uso de Focal Loss adaptada con kernel Gaussiano. 2) Rama Box Vertices: uso de MDL optimizada seleccionando el valor mínimo entre cuatro permutaciones cíclicas de vértices (Eq. 4), solucionando la discontinuidad angular. 3) Rama Offset: predicción de desplazamientos sub-píxel utilizando la distancia de Mahalanobis adaptada al tamaño de la caja (Eq. 5).

### III. Experiments
Evaluación experimental empírica de MDL en el dataset DOTA-v1.0 y comparación mediante estudios de ablación y con detectores del estado del arte.
* **Problemas atacados**: Validación práctica de la precisión y estabilidad de MDL sobre una implementación de red real.
* **Limitaciones de ese entonces**: Falta de pruebas exhaustivas que contrastaran los beneficios de calcular la covarianza sobre las coordenadas estimadas frente a las de referencia (ground truth).
* **Soluciones alcanzadas**: Protocolo de entrenamiento usando optimizador Adam, aceleración sobre múltiples GPUs RTX 3090, preprocesamiento por parches de $600 \times 600$ y fusión multiescala por NMS.

#### A. Dataset
Especificaciones del dataset DOTA-v1.0.
* **Problemas atacados**: Ajustar las imágenes aéreas al tamaño de entrada del detector.
* **Limitaciones de ese entonces**: Las imágenes gigantes de DOTA superan la capacidad de memoria del modelo CenterNet.
* **Soluciones alcanzadas**: División sistemática en parches de $600 \times 600$ píxeles con un solape de 100 píxeles a escalas de 0.5 y 1.0, generando conjuntos de entrenamiento y test densos.

#### B. Implementation and Testing Details
Configuración de hiperparámetros de entrenamiento y criterios de inferencia.
* **Problemas atacados**: Optimización convergente del detector y umbrales de predicción óptimos.
* **Limitaciones de ese entonces**: El decaimiento abrupto de la tasa de aprendizaje puede provocar atascos en mínimos locales.
* **Soluciones alcanzadas**: Uso del optimizador Adam con una tasa de aprendizaje de $1.25 \times 10^{-4}$ acoplada a un planificador de decaimiento exponencial del 0.96% por epoch, y selección de las top-500 propuestas de centro con confianza superior a 0.1 en inferencia.

#### C. Ablation Study
Ablación de las variantes de covarianza de MDL y comparación directa contra Smooth L1 Loss.
* **Problemas atacados**: Seleccionar la formulación óptima de la matriz de covarianza $\Sigma$ y medir la ganancia frente a pérdidas clásicas.
* **Limitaciones de ese entonces**: No existía consenso sobre si estimar la covarianza sobre las predicciones de la red (MDL-p) o sobre el ground truth fijo (MDL-t).
* **Soluciones alcanzadas**: Las pruebas revelan que: 1) MDL-p (76.16% mAP) supera a MDL-t (74.33% mAP), indicando que calcular la covarianza sobre las predicciones móviles facilita dinámicamente la regresión. 2) MDL-p supera a Smooth L1 (73.98% mAP) por 2.18 puntos de precisión mAP y exhibe curvas de optimización mucho más estables y libres de picos (Fig. 6).

#### D. Further Comparison
Comparativa exhaustiva con detectores de 5 y 8 parámetros en las 15 categorías de DOTA.
* **Problemas atacados**: Posicionar el método MDL frente a técnicas líderes de regresión orientada.
* **Limitaciones de ese entonces**: Métodos de 8 parámetros previos requerían parametrizaciones complejas e indirectas (ej. BBAVectors que predice vectores adicionales de horizontalidad).
* **Soluciones alcanzadas**: MDL-p alcanzó un mAP récord del 76.16%, superando a métodos consolidados como RoI Transformer (72.8%), SCRDet (72.61%), RSDet (74.1%) y BBAVectors (75.36%), y liderando de forma absoluta en categorías difíciles como Puertos, Helicópteros y Vehículos Grandes.

### IV. Conclusion
Resumen de las contribuciones de MDL y propuestas de trabajo futuro.
* **Problemas atacados**: Delinear el uso de pérdidas invariantes de escala en visión por computadora para polígonos complejos.
* **Limitaciones de ese entonces**: La discontinuidad de frontera angular sigue requiriendo soluciones de permutación ad-hoc en lugar de resolverse en la formulación de pérdida misma.
* **Soluciones alcanzadas**: Consolidación de MDL como una alternativa robusta a las pérdidas Minkowski, proponiendo para el futuro su extensión hacia la detección de cuadriláteros o polígonos irregulares generales (conjuntos de puntos mayores a 4).
