# Faster CNN-based vehicle detection and counting strategy for fixed camera scenes

- **Key**: Gomaa2022
- **Year**: 2022
- **Venue**: Multimedia Tools and Applications

## Resumen
Este artículo presenta un enfoque eficiente y en tiempo real para la detección y el conteo de vehículos en movimiento capturados por cámaras de videovigilancia fijas (CCTV). El sistema aborda las limitaciones de velocidad y precisión de los métodos convencionales al proponer una estrategia híbrida que fusiona la detección basada en redes neuronales convolucionales (YOLOv2) con el análisis cinemático de puntos de características viales. El pipeline opera en dos etapas: (1) detección de vehículos y (2) conteo de trayectorias en movimiento. Para minimizar el costo computacional de ejecutar la red profunda en cada fotograma, el detector YOLOv2 (utilizando ResNet-50 como backbone y ajustado mediante transfer learning) se aplica únicamente en el primer fotograma de cada conjunto de fotogramas (frameset) de tamaño fijo $N=10$. Durante los fotogramas restantes, la posición de los vehículos se propaga usando el algoritmo de flujo óptico Kanade-Lucas-Tomasi (KLT), refinando las cajas delimitadoras con agrupamiento K-means para suprimir falsos positivos del fondo. En la etapa de conteo, se calcula la superposición espacial (con un umbral $\alpha=25\%$) entre las trayectorias seguidas y las nuevas cajas detectadas para asignar IDs únicos y evitar conteos redundantes. Los experimentos en 12 videos de bases de datos públicas (GRAM, CDnet2014, UA-DETRAC, ATON) demuestran que el sistema alcanza 18.7 FPS de velocidad promedio en CPU, superando en velocidad a Faster R-CNN (1.24 FPS) y a redes de sustracción de fondo basadas en CNN (BS-CNN) (0.19 FPS), a la vez que incrementa el recall promedio en un 5.5% gracias al flujo de información bidireccional entre el detector y el rastreador.

## Secciones y Subsecciones

### 1 Introduction
Introduce el rol del análisis del flujo de tráfico en los sistemas de transporte inteligentes (ITS) para mitigar la congestión vial, planificar rutas y estimar densidades vehiculares.
* **Problemas atacados**: Diseñar un sistema de conteo automático de vehículos en flujos de video que sea preciso, tolerante a variaciones climáticas y de luz, y que a la vez sea lo suficientemente rápido para operar en tiempo real en infraestructura comercial.
* **Limitaciones de ese entonces**: Los sensores físicos intrusivos tienen un alto costo de instalación. Las técnicas clásicas de visión artificial (sustracción de fondo, GMM, PCA) son sensibles a sombras y rotaciones. Los detectores de aprendizaje profundo modernos (como Faster R-CNN) logran una excelente precisión pero son computacionalmente costosos, operando a menos de 2 FPS en flujos continuos. Por último, la mayoría de sistemas maneja la detección y el tracking de forma aislada sin retroalimentación.
* **Soluciones alcanzadas**: Se propone un esquema colaborativo donde el detector y el tracker operan de forma síncrona. YOLOv2 realiza la inicialización espacial de cajas y KLT propaga y depura estas detecciones a lo largo de un frameset de 10 fotogramas.

### 2 Related work
Revisión del estado del arte en redes de clasificación de imágenes profundas (DCNN), detectores de dos etapas y detectores de una sola etapa.
* **Problemas atacados**: La necesidad de optimizar el compromiso (trade-off) entre la precisión de localización y la velocidad de inferencia de los detectores viales.
* **Limitaciones de ese entonces**: Modelos como RCNN, Fast R-CNN y Faster R-CNN son muy precisos pero lentos debido a la generación secuencial de propuestas de región. Los modelos monofásicos rápidos (YOLOv2, YOLOv3, YOLOv4, SSD) mejoran la velocidad pero predecir frame a frame de forma aislada sigue siendo costoso y genera fluctuaciones temporales (detecciones perdidas y falsos positivos temporales). Los sistemas basados en ORB+RANSAC o DeepSORT mejoran el tracking pero agregan cuellos de botella temporales (p. ej., RANSAC estimando matrices homográficas consume mucho cómputo).
* **Soluciones alcanzadas**: Se demuestra la viabilidad de usar YOLOv2 acoplado con flujo óptico liviano KLT y agrupamiento espectral/K-means para refinar el ruido visual sin introducir redes adicionales de deep tracking pesadas.

### 3 Proposed methodology
Sintetiza las fases y el flujo arquitectónico del sistema distribuido por conjuntos de fotogramas (framesets).
* **Problemas atacados**: Estructurar la lógica de comunicación entre el detector de regresión YOLOv2 y el propagador óptico temporal.
* **Limitaciones de ese entonces**: La pérdida de información de objetos estáticos o parados en el tráfico que escapan a los filtros de movimiento tradicionales.
* **Soluciones alcanzadas**: El video se procesa en bloques o framesets de longitud $N=10$. En el primer fotograma se invoca a YOLOv2; en los siguientes 9 fotogramas, se extraen y propagan puntos de características morfológicas, agrupando las trayectorias resultantes para asignar etiquetas únicas de conteo.

#### 3.1 Vehicle detection
* **Problemas atacados**: Clasificación rápida y delimitación espacial aproximada de los vehículos en la escena vial.
* **Limitaciones de ese entonces**: La baja precisión y alto costo de tiempo de los extractores de parches visuales clásicos y de los métodos de sustracción de fondo aplicados a píxeles.
* **Soluciones alcanzadas**: Se aplica transfer learning sobre un modelo YOLOv2 preentrenado. Se reemplaza la capa fully connected final de 1000 clases (ImageNet) por una capa de dos clases (vehículo vs. fondo). Se emplea ResNet-50 como backbone extractor. Las detecciones iniciales se derivan a la fase de tracking para descartar falsos positivos viales (que poseen dinámicas espaciales diferentes al tráfico real).

#### 3.2 Vehicle features refinement and clustering
* **Problemas atacados**: Depurar detecciones erróneas generadas por el detector YOLOv2 y unificar múltiples puntos de interés en un solo vehículo.
* **Limitaciones de ese entonces**: YOLOv2 puede generar falsos positivos en regiones estáticas de fondo (como árboles o edificios) y ruido de sombras.
* **Soluciones alcanzadas**: Se calcula el flujo óptico de Kanade-Lucas (KLT) para derivar vectores de velocidad y dirección (ángulo $\theta_i$) de los puntos característicos. Solo se valida una caja como vehículo si sus puntos característicos se mantienen rastreables de forma coherente durante al menos 9 fotogramas consecutivos (eliminando tracks ruidosos efímeros). Los puntos supervivientes se agrupan espacialmente mediante K-means para definir los límites corregidos de los vehículos en la escena.

#### 3.3 Vehicle counting
* **Problemas atacados**: Asignación inequívoca de identificadores de trayectoria y conteo exacto de los vehículos.
* **Limitaciones de ese entonces**: Las oclusiones mutuas temporales o las detecciones perdidas en frames individuales duplican el conteo del mismo vehículo físico al reaparecer.
* **Soluciones alcanzadas**: Se rastrean los centroides de los clusters vehiculares asignándoles un ID único. Para asociar las nuevas detecciones al final del frameset, se mide el área de intersección espacial entre la caja rastreada y la detectada. Si supera un umbral $\alpha=25\%$, se mantiene el ID. Se definen 4 estados lógicos de conteo para asegurar que solo los vehículos nuevos incrementen el contador en uno y los existentes se propaguen sin duplicarse.

### 4 Experimental results
Evaluación experimental del sistema en 12 secuencias de video públicas bajo desafíos de iluminación diurna/nocturna, oclusiones y congestión.
* **Problemas atacados**: Cuantificar el desempeño del detector YOLOv2 entrenado bajo diferentes distribuciones y medir el impacto cinemático del tracking.
* **Limitaciones de ese entonces**: Evaluar los detectores de deep learning en datos de prueba muy diferentes a los de entrenamiento provoca un colapso en la exactitud.
* **Soluciones alcanzadas**: Se diseñan tres experimentos de entrenamiento progresivo. El Experimento I (entrenamiento en Matlab, 295 imágenes) arrojó falsos negativos en escenarios reales. El Experimento II (entrenamiento con 127 fotogramas de CDnet) mejoró el desempeño local pero no generalizó en otros datasets. El Experimento III (entrenamiento híbrido con menos del 10% de frames de varias escenas CDnet y Matlab) logró detectar exitosamente vehículos en escenas nocturnas complejas y de alta congestión.

El modelo propuesto YOLOv2+KLT alcanzó una velocidad de 18.7 FPS en promedio, superando drásticamente a Faster R-CNN (1.24 FPS) y BS-CNN (0.19 FPS). Asimismo, al comparar el modelo YOLOv2 operando frame a frame sin tracking frente a la propuesta híbrida (YOLOv2+KLT con detección cada $N=10$ frames), se demostró que el tracking reduce el costo computacional elevando la tasa de FPS en un 90% en promedio. Además, el seguimiento ayudó a recuperar vehículos que el detector YOLOv2 omitió temporalmente, elevando el recall de detección en un 5.5% en promedio en las secuencias Highway, M-30 y Highway II.

### 5 Conclusion
Sintetiza las aportaciones y discute la viabilidad del pipeline propuesto.
* **Problemas atacados**: Validar la superioridad del modelo propuesto en términos de velocidad de procesamiento y precisión de conteo.
* **Limitaciones de ese entonces**: YOLOv2 requiere un entrenamiento multiescena exhaustivo para evitar fallos por Domain Shift cuando se despliega en ángulos de cámara desconocidos.
* **Soluciones alcanzadas**: Se valida que la combinación de YOLOv2 y análisis de puntos de flujo óptico KLT constituye una solución de bajo costo y alta velocidad (18.7 FPS) que supera las deficiencias de los clasificadores basados en píxeles y los algoritmos tradicionales de ITS.
