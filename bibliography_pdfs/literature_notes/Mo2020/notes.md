# Improved Faster RCNN Based on Feature Amplification and Oversampling Data Augmentation for Oriented Vehicle Detection in Aerial Images

- **Key**: Mo2020
- **Year**: 2020
- **Venue**: Remote Sensing

## Resumen
El artículo propone un marco de detección de vehículos orientados en imágenes aéreas basado en una versión mejorada de Faster R-CNN. Los vehículos en imágenes aéreas presentan dos retos principales: tamaño muy pequeño (típicamente menor a 32x32 píxeles) y una distribución de categorías altamente desbalanceada (foreground-foreground class imbalance). Para resolver el desbalance, se propone una estrategia de aumento de datos basada en sobremuestreo y costura (*oversampling and stitching data augmentation*), equilibrando el número de muestras por categoría. Para mitigar la pérdida de resolución espacial en objetos pequeños producida por las sucesivas capas de pooling en redes profundas (como ResNet-101), se propone un módulo de amplificación de características (*feature amplification*) por interpolación bilineal en el último mapa de características. Finalmente, para aumentar la discriminación entre clases similares y manejar la orientación aleatoria de los vehículos, se diseña una función de pérdida conjunta que incorpora pérdidas por regresión de cajas horizontales (HBB) y orientadas (OBB), junto con la pérdida de centro (*center loss*). Evaluado en el dataset VEDAI, el método logra una mejora de aproximadamente 8% en mAP comparado con Faster R-CNN clásico.

## Secciones y Subsecciones

### 1. Introducción
Presenta la importancia de la detección orientada de vehículos en imágenes de satélite o aéreas de alta resolución para tráfico y conducción autónoma. Define las dificultades asociadas a objetos pequeños e identifica las debilidades del paradigma Faster R-CNN estándar en este dominio.
* **Problemas atacados**: La baja precisión y el fallo en la clasificación de vehículos pequeños con orientaciones arbitrarias y clases desbalanceadas en fotografía aérea.
* **Limitaciones de ese entonces**: Los detectores 2D clásicos asumen cajas horizontales (HBB) que no describen adecuadamente la orientación del vehículo y mezclan información de fondo. Los modelos sufren de sobreajuste hacia clases mayoritarias (como automóviles) y fallan en detectar clases minoritarias (como aviones o tractores). Además, el submuestreo de las capas convolucionales reduce objetos pequeños a pocos píxeles en las últimas capas, perdiendo detalles clave para la clasificación.
* **Soluciones alcanzadas**: Formulación de un framework integrado basado en Faster R-CNN con aumento de datos por sobremuestreo, interpolación bilineal para amplificación de características y pérdida multi-tarea con center loss y cajas orientadas (OBB).

### 2. Trabajo Relacionado
Revisa la literatura científica asociada a los tres retos identificados.

#### 2.1. Problema de Desbalance de Clases
Analiza las soluciones previas al desbalance entre clases del primer plano (foreground-foreground imbalance).
* **Problemas atacados**: El sesgo de entrenamiento en redes convolucionales hacia las categorías con mayor número de muestras.
* **Limitaciones de ese entonces**: Los métodos de balanceo de clases (como muestreo balanceado o intercambio de muestras) se diseñaron principalmente para imágenes naturales terrestres. En teledetección, el desbalance ha sido poco explorado, limitando el éxito de los modelos en categorías raras.
* **Soluciones alcanzadas**: Propuesta de una técnica de segmentación de plantillas de vehículos para sobremuestreo y costura sintética en imágenes de fondo reales.

#### 2.2. Representación de Objetos Pequeños
Examina los enfoques para enriquecer detalles de objetos con dimensiones menores a 32x32 píxeles.
* **Problemas atacados**: La pérdida de información de grano fino debido a las operaciones de pooling en CNNs profundas.
* **Limitaciones de ese entonces**: Las soluciones tradicionales aumentan la resolución de la imagen (súper-resolución) o construyen pirámides de imágenes/características complejas (FPN). Esto eleva drásticamente el costo de memoria de la GPU, los parámetros de red y el retardo en tiempo real.
* **Soluciones alcanzadas**: Propuesta de amplificar directamente el mapa de características profundo utilizando una interpolación bilineal simple y ligera, preservando la semántica de la red sin añadir parámetros aprendidos.

#### 2.3. Capacidad de Discriminación de las Características
Analiza los métodos para aumentar la separación de clases visualmente similares en teledetección.
* **Problemas atacados**: El error de clasificación entre subtipos de vehículos (vans, pick-ups, camiones, etc.) debido a la alta similitud inter-clase vista desde el cenit.
* **Limitaciones de ese entonces**: El uso de características contextuales o concatenación directa de capas tempranas y tardías ayuda a la localización pero no penaliza la dispersión de características del mismo tipo en el espacio latente.
* **Soluciones alcanzadas**: Introducción de la pérdida de centro (*center loss*) para minimizar las distancias intra-clase en el espacio de características.

#### 2.4. Detección de Objetos Orientados
Describe los desarrollos en regresión de cajas orientadas (OBB).
* **Problemas atacados**: La imprecisión de las cajas horizontales al encuadrar vehículos orientados diagonalmente, lo que añade ruido de fondo e interfiere en el NMS.
* **Limitaciones de ese entonces**: La mayoría de los detectores solo producen HBB. Pocos frameworks implementan de forma coordinada e integrada predicciones de cajas horizontales y orientadas en paralelo.
* **Soluciones alcanzadas**: Diseño de un cabezal de regresión de 5 variables $(x, y, w, h, \theta)$ que se entrena de forma conjunta con el cabezal HBB estándar.

### 3. Materiales y Métodos
Detalla el diseño del framework propuesto.

#### 3.1. Arquitectura General
Presenta el flujo de trabajo global basado en Faster R-CNN con backbone ResNet-101.
* **Problemas atacados**: Integrar armónicamente el aumento de datos, la amplificación de mapas de características profundos y la pérdida multi-tarea.
* **Limitaciones de ese entonces**: Las modificaciones ad-hoc a Faster R-CNN a menudo degradan una métrica al optimizar otra (p. ej., mejorar OBB pero dañar la clasificación).
* **Soluciones alcanzadas**: Estructura de tres etapas: (1) Aumento por sobremuestreo y costura en el dataset de entrada; (2) Extracción en ResNet-101 seguida de amplificación bilineal en el mapa final; (3) Cabezales paralelos de clasificación y regresión HBB/OBB entrenados con center loss.

#### 3.2. Aumento de Datos para el Desbalance de Clases mediante Sobremuestreo y Costura
Explica el procedimiento de síntesis de imágenes balanceadas.
* **Problemas atacados**: La escasez de muestras en categorías minoritarias en el dataset VEDAI (como Plane o Tractor frente a Car).
* **Limitaciones de ese entonces**: El aumento clásico (rotaciones, reflejos) no altera la proporción relativa de las categorías, manteniendo el sesgo del optimizador.
* **Soluciones alcanzadas**: Algoritmo en 6 pasos: (1) Rotación de imágenes originales a 90°, 180° y 270°; (2) Recorte de vehículos para crear un catálogo de plantillas; (3) Identificación de la clase mayoritaria como benchmark de expansión; (4) Selección de imágenes de fondo con baja densidad de vehículos; (5) Costura aleatoria de plantillas minoritarias en fondos asegurando un solapamiento (overlap) de cero; (6) Repetición hasta equilibrar el conteo de todas las clases.

#### 3.3. Amplificación de Características Profundas para Objetos Pequeños
Explica el uso de la interpolación bilineal sobre el mapa de características final del backbone.
* **Problemas atacados**: La pérdida extrema de información espacial de vehículos pequeños. Si un vehículo de 32x32 píxeles pasa por las 4 etapas de pooling de ResNet-101, su representación en el mapa de salida se reduce a solo 2x2 píxeles, lo que resulta insuficiente para extraer descriptores de clase.
* **Limitaciones de ese entonces**: Las convoluciones de deconvolución para upsampling a menudo introducen artefactos de tablero de ajedrez (*checkerboard artifacts*) que degradan la calidad del descriptor.
* **Soluciones alcanzadas**: Amplificación del mapa de características final por un factor multiplicativo (el mejor resultado se obtiene con un factor de 2.0) usando interpolación bilineal. Esto expande el mapa de $2\times2$ a $4\times4$ píxeles de manera suave, recuperando detalles finos sin distorsiones geométricas.

#### 3.4. Función de Pérdida Multi-Tarea para Cajas Horizontales y Orientadas Conjuntas
Establece la formulación matemática de la optimización del modelo.
* **Problemas atacados**: Optimizar simultáneamente la exactitud de la forma (HBB), la orientación (OBB) y la precisión de clase (center loss).
* **Limitaciones de ese entonces**: La pérdida tradicional de Faster R-CNN (Cross Entropy + Smooth L1) no restringe la distancia entre las características de una misma clase, resultando en clasificaciones erróneas entre vehículos de silueta similar.
* **Soluciones alcanzadas**: Formulación de una pérdida multi-tarea de 5 componentes: pérdidas de clasificación y regresión HBB, pérdidas de clasificación y regresión OBB (usando 5 parámetros offset respecto a anclajes horizontales), y la pérdida de centro (*center loss*) que penaliza la distancia euclidiana entre las características latentes del lote y el centroide de su clase.

### 4. Resultados Experimentales y Configuración
Describe los experimentos empíricos y análisis de ablación.

#### 4.1. Descripción de Datasets y Detalles de Implementación
Presenta el dataset VEDAI y los parámetros del hardware y entrenamiento.
* **Problemas atacados**: Evaluar la robustez del método bajo directrices estandarizadas.
* **Limitaciones de ese entonces**: Variaciones en las particiones de entrenamiento y prueba pueden adulterar las comparaciones de mAP.
* **Soluciones alcanzadas**: Uso de VEDAI (imágenes de 1024x1024 píxeles, 12.5 cm de GSD), partición aleatoria de 50/50 y repetición de entrenamientos para reducir variaciones. Configuración en TensorFlow sobre GPU GTX 1080Ti, 100k iteraciones, y anclajes con 5 escalas y 13 relaciones de aspecto para cubrir variados tamaños de vehículos.

#### 4.2. Resultados de Detección y Comparación con Líneas Base
Compara la precisión mAP del método contra Faster R-CNN-HA, FPN-RA, DFPN-RA, FPN-HA y DFPN-HA.
* **Problemas atacados**: Demostrar el beneficio de usar anclajes horizontales en lugar de rotados y la superioridad del modelo mejorado.
* **Limitaciones de ese entonces**: Se creía que usar anclajes rotados (RA) era indispensable para detectar objetos inclinados.
* **Soluciones alcanzadas**: Se demuestra que los anclajes rotados reducen la información de contexto en el ROI pooling, bajando la precisión. El método propuesto supera a todas las líneas base, logrando 60.4% mAP en HBB y 60.1% mAP en OBB (un incremento neto de ~8% sobre Faster R-CNN estándar).

#### 4.3. Comparación de Diferentes Métodos de Aumento de Datos
Compara los datasets generados por rotación (R), sobremuestreo (O) y la unión de ambos (M).
* **Problemas atacados**: Analizar el impacto del balance de clases en el entrenamiento del detector.
* **Limitaciones de ese entonces**: El sobremuestreo sintético puro puede distorsionar el contexto real de la imagen.
* **Soluciones alcanzadas**: El dataset de sobremuestreo (O) supera al de rotaciones puras (R) al aumentar el recall de clases difíciles. La combinación mixta (M) obtiene los mejores resultados al dotar a la red de un balance de clases y una alta diversidad de contextos geográficos reales.

#### 4.4. Estudio de Ablación
Analiza de forma aislada el aporte de cada componente tecnológico en la precisión final.
* **Problemas atacados**: Desglosar cuantitativamente el origen de la mejora del mAP.
* **Limitaciones de ese entonces**: Dificultad para demostrar qué modificaciones (datos, mapas o pérdidas) son realmente efectivas.
* **Soluciones alcanzadas**: 
  - La amplificación de características bilineal aporta un incremento de 3% mAP.
  - El balanceo del dataset aporta 3% mAP (HBB) y 5% mAP (OBB).
  - La center loss incrementa la precisión en un 2% mAP al mitigar la confusión entre tipos de vehículos similares.

### 5. Discusión
Analiza cuantitativamente la densidad de muestras y la sintonía del hiperparámetro de amplificación.

#### 5.1. Análisis del Número de Objetos y Muestras Positivas
* **Problemas atacados**: Validar la hipótesis de que el sobremuestreo mejora el ajuste de los anclajes RPN.
* **Limitaciones de ese entonces**: Falta de evidencia directa de que la síntesis de datos beneficia al RPN en clases minoritarias.
* **Soluciones alcanzadas**: Demostración de que el sobremuestreo aumenta drásticamente el número de anclajes positivos (IoU > 0.7) en el entrenamiento del RPN para categorías difíciles (p. ej., en la clase Plane, el número de muestras positivas pasa de casi cero a niveles comparables con la clase Car).

#### 5.2. Análisis de los Parámetros de Amplificación de Características
Estudia la variación del factor de escala y el tipo de interpolación.
* **Problemas atacados**: Determinar el nivel de escala óptimo y el mejor algoritmo matemático de escalado.
* **Limitaciones de ese entonces**: El escalado excesivo puede introducir ruido y la interpolación por vecino más cercano causa distorsiones de bordes.
* **Soluciones alcanzadas**: La interpolación bilineal con un factor de 2.0 es óptima (logrando 60.7% mAP en HBB y 60.4% mAP en OBB). Factores mayores (2.5 o 3.0) o menores (1.5) reducen el rendimiento. La interpolación de vecino más cercano (NN-2.0) empeora la precisión incluso respecto al modelo sin amplificar debido a la pixelación y artefactos tipo diente de sierra.

### 6. Conclusiones
Resume las aportaciones del trabajo y delinea las futuras líneas de investigación.
* **Problemas atacados**: La persistencia de falsos positivos debido a confusión con objetos de fondo en teledetección.
* **Limitaciones de ese entonces**: A pesar de las mejoras en mAP, la tasa de precisión pura sigue viéndose afectada por objetos del paisaje similares a vehículos.
* **Soluciones alcanzadas**: Validación de la efectividad del framework propuesto y propuesta de estudiar en el futuro descriptores de contexto más potentes para discriminar falsos positivos en objetos aéreos de escala ultra-pequeña.
