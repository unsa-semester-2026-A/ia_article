# Object Detection in 20 Years: A Survey

- **Key**: Zou2023ObjectDetectionSurvey
- **Year**: 2023
- **Venue**: Proceedings of the IEEE

## Resumen
Este artículo presenta una revisión exhaustiva y detallada de la evolución de la detección de objetos a lo largo de un período de un cuarto de siglo (desde finales de la década de 1990 hasta 2022). La detección de objetos es un pilar fundamental en la visión por computadora que busca responder a la pregunta fundamental de qué objetos de interés están en qué lugares de la imagen digital. El artículo divide la historia del área en dos eras principales: la era de la detección tradicional basada en características artesanales (antes de 2014) y la era del aprendizaje profundo (después de 2014). A través de esta taxonomía, los autores discuten hitos clave (como Viola-Jones, HOG, DPM, R-CNN, YOLO y DETR), examinan conjuntos de datos icónicos (PASCAL VOC, ImageNet, MS COCO y Open Images) y analizan críticamente la evolución técnica en la escala espacial, la incorporación de contexto, la minería de ejemplos negativos difíciles, las funciones de pérdida y las estrategias de post-procesamiento (NMS). Asimismo, se ofrece una revisión detallada de las tecnologías de aceleración para despliegues eficientes en tiempo real y una proyección hacia las direcciones futuras del área.

## Secciones y Subsecciones

### I. Introduction
Se presenta una introducción general a la detección de objetos y su papel como base para tareas visuales más complejas (segmentación de instancias, subtitulado de imágenes y seguimiento). Define las métricas principales de desempeño: velocidad y precisión.
* **Problemas atacados**: Identificar y clasificar de forma automatizada múltiples objetos en imágenes digitales, superando retos como oclusiones, variaciones extremas de escala e iluminación y cambios de orientación.
* **Limitaciones de ese entonces**: Históricamente, las soluciones carecían de generalización, sufriendo cuellos de botella severos cuando el entorno de la imagen variaba levemente.
* **Soluciones alcanzadas**: Sistematización de la evolución tecnológica y establecimiento de este survey como una hoja de ruta conceptual para investigadores y desarrolladores que aborda la historia, aceleración y fronteras de la detección.

### II. Object Detection in 20 Years
Estudio cronológico y técnico de los hitos algorítmicos, bases de datos y métricas que marcaron la historia de la detección de objetos en las últimas dos décadas.
* **Problemas atacados**: Organizar la vasta literatura del área en una estructura temporal y evolutiva clara, destacando las ideas fundamentales detrás de los saltos de rendimiento.
* **Limitaciones de ese entonces**: Las revisiones bibliográficas previas solían centrarse en detalles específicos de implementación de modelos concretos, ignorando la evolución conceptual de los componentes modulares.
* **Soluciones alcanzadas**: División formal en la "Era Tradicional" y la "Era del Aprendizaje Profundo" (Fig. 2), detallando las tecnologías que impulsaron la precisión en benchmarks clave (Fig. 3).

#### A. A Road Map of Object Detection
Descripción detallada del camino evolutivo y los modelos hitos en la detección de objetos.
* **Problemas atacados**: Categorizar el progreso desde algoritmos con ventanas deslizantes manuales hasta modelos basados en transformadores sin anclajes.
* **Limitaciones de ese entonces**: Falta de representaciones de características genéricas capaces de capturar la semántica de múltiples clases simultáneamente.
* **Soluciones alcanzadas**: Agrupación analítica de los detectores en tradicionales, de dos etapas (two-stage) y de una sola etapa (one-stage), proporcionando sus pautas de diseño y contribuciones.

##### 1) Milestones: Traditional Detectors
* **Problemas atacados**: Procesamiento en tiempo real bajo severas restricciones de cómputo y el modelamiento de la deformabilidad de los objetos.
* **Limitaciones de ese entonces**: Falta de aceleradores de hardware y de descriptores robustos. El Viola-Jones (VJ, 2001) resolvió la lentitud de la ventana deslizante mediante imágenes integrales, cascadas de rechazo y selección por AdaBoost. HOG (2005) representó la forma peatonal de manera densa pero sufría ante deformaciones severas del cuerpo.
* **Soluciones alcanzadas**: Consolidación del modelo DPM (Deformable Part-based Model, 2008) como el estándar de oro tradicional, el cual implementa el concepto de "divide y vencerás" mapeando objetos como conjuntos de partes interconectadas mediante resortes virtuales (star model y mixture model) e introduce la regresión de cajas y minería de negativos difíciles.

##### 2) Milestones: CNN based Two-stage Detectors
* **Problemas atacados**: Superar la saturación de los descriptores artesanales (HOG, LBP) integrando representaciones de aprendizaje profundo y optimizando la velocidad de dos etapas.
* **Limitaciones de ese entonces**: R-CNN (2014) revolucionó el área usando propuestas de región y CNNs pero era extremadamente lento (14s por imagen) ya que extraía características redundantes para más de 2000 propuestas de forma aislada.
* **Soluciones alcanzadas**: Evolución secuencial: 1) SPPNet (2014) introduce la capa de pooling espacial piramidal que permite procesar la imagen completa una sola vez. 2) Fast R-CNN (2015) unifica la clasificación y regresión de cajas bajo una pérdida conjunta entrenable extremo a extremo. 3) Faster R-CNN (2015) introduce el Region Proposal Network (RPN) sustituyendo algoritmos externos de propuesta y 4) FPN (2017) introduce la pirámide de características semánticas por conexiones laterales.

##### 3) Milestones: CNN based One-stage Detectors
* **Problemas atacados**: Lograr inferencia en tiempo real en una sola etapa sin sacrificar severamente la precisión en objetos pequeños y densos.
* **Limitaciones de ese entonces**: YOLOv1 (2015) era veloz (45 FPS) pero sufría de baja precisión de localización en objetos agrupados. SSD (2015) mejoró la escala evaluando múltiples capas de resolución pero no solucionaba el problema de desbalance extremo entre fondo y primer plano.
* **Soluciones alcanzadas**: 1) RetinaNet (2017) introduce Focal Loss resolviendo el desbalance de clases de detectores densos. 2) CornerNet (2018) y CenterNet (2019) formulan la tarea como detección de puntos clave (esquinas y centros), eliminando la dependencia de anclajes (anchors). 3) DETR (2020) introduce Transformers con correspondencia de conjuntos bipartitos eliminando anclajes y NMS en inferencia.

#### B. Object Detection Datasets and Metrics
Análisis de los conjuntos de datos que han guiado la investigación y los estándares de evaluación.
* **Problemas atacados**: Estandarizar la medición del rendimiento y la robustez de los algoritmos de detección.
* **Limitaciones de ese entonces**: Benchmarks tempranos medían la tasa de fallos sobre ventanas de prueba aisladas (FPPW), lo cual era defectuoso y no predecía el desempeño real sobre imágenes completas.
* **Soluciones alcanzadas**: Adopción de la métrica de Precisión Promedio (AP) al umbral IoU=0.5 introducida en PASCAL VOC y, posteriormente, del AP promedio ponderado en múltiples umbrales (0.5 a 0.95) en MS COCO que exige una alta exactitud espacial de localización.

##### 1) Datasets
* **Problemas atacados**: Minimizar el sesgo de los conjuntos de datos y entrenar redes con alta capacidad de generalización.
* **Limitaciones de ese entonces**: Datasets iniciales (como VOC07/12) eran pequeños (menos de 11k imágenes) y limitados a pocas clases.
* **Soluciones alcanzadas**: Creación de bases de datos masivas multisectoriales (Table I) como ImageNet (200 clases), MS COCO (80 clases, segmentación de instancias) y Open Images (600 clases, 15 millones de cajas), permitiendo la transición a modelos de alta escala.

##### 2) Metrics
* **Problemas atacados**: Medir con precisión la fidelidad geométrica y la coincidencia de las detecciones de la red.
* **Limitaciones de ese entonces**: Métricas FPPW obsoletas.
* **Soluciones alcanzadas**: Estandarización de la métrica mAP combinada con el umbral IoU (Intersection over Union). La adopción del AP ponderado de MS COCO impulsó la precisión de localización en aplicaciones críticas (robótica, conducción autónoma).

#### C. Technical Evolution in Object Detection
Análisis temático y detallado del desarrollo evolutivo de los componentes modulares clave de los detectores.
* **Problemas atacados**: Comprender cómo el área ha resuelto históricamente los cinco retos principales: escala, contexto, desbalance de muestras, optimización de pérdidas y post-procesamiento.
* **Limitaciones de ese entonces**: Falta de análisis unificado de estos cinco módulos, lo cual fragmentaba el entendimiento conceptual del diseño de detectores.
* **Soluciones alcanzadas**: Mapeo y análisis de la evolución técnica individual de cada componente para guiar el desarrollo de detectores modernos.

##### 1) Technical Evolution of Multi-Scale Detection
* **Problemas atacados**: Detectar de forma simultánea objetos diminutos (autos lejanos) y gigantes (edificios o puentes).
* **Limitaciones de ese entonces**: El escalado tradicional de imágenes y ventanas de deslizamiento es lento y consume gran memoria.
* **Soluciones alcanzadas**: Transición conceptual en 5 períodos (Fig. 5): 1) pirámides de imágenes con ventanas deslizantes, 2) propuestas de objetos agnósticas de clase, 3) regresión directa profunda libre de anclajes (YOLO), 4) detección multirreferencia (anchors de Faster R-CNN) y 5) detección multirresolución utilizando pirámides de características en la red (SSD, FPN, Cascade R-CNN).

##### 2) Technical Evolution of Context Priming
* **Problemas atacados**: Aprovechar las relaciones semánticas del entorno para mejorar la clasificación y localización de objetos difíciles.
* **Limitaciones de ese entonces**: Detectores tradicionales solo evaluaban la ventana local del objeto, perdiendo valiosas asociaciones de contexto terrestre.
* **Soluciones alcanzadas**: Evolución en 3 etapas (Fig. 6): 1) Contexto Local: ampliar el campo de visión del RoI (GBDNet, CoupleNet), 2) Contexto Global: resúmenes estadísticos de escena (Gist) o convoluciones deformables y mapas de atención total (Non-local, DETR) y 3) Interacciones de Contexto: modelar las relaciones inter-objeto mediante redes de relación recurrentes o de grafos (RelationNet, RescoringNet).

##### 3) Technical Evolution of Hard Negative Mining
* **Problemas atacados**: Superar el desbalance extremo de muestras en detectores densos donde la proporción de fondo a objetos puede ser de $10^7 : 1$.
* **Limitaciones de ese entonces**: El cálculo ingenuo sobre todo el fondo inunda el gradiente con ejemplos fáciles que ahogan el aprendizaje de características discriminativas.
* **Soluciones alcanzadas**: Transición histórica (Fig. 7): 1) Bootstrap tradicional: iniciar con un conjunto pequeño e incorporar iterativamente falsos positivos detectados en DPM. 2) Descarte temporal en CNNs (2014-2016) confiando en pesos equilibrados simples y 3) Re-introducción avanzada después de 2016 mediante OHEM (Online Hard Example Mining) y el diseño de pérdidas dedicadas como Focal Loss.

##### 4) Technical Evolution of Loss Function
* **Problemas atacados**: Definir la supervisión óptima para la clasificación categórica y la coincidencia de las coordenadas de las cajas.
* **Limitaciones de ese entonces**: El uso de pérdidas MSE (L2) para la localización trata los cuatro componentes $(x, y, w, h)$ como independientes, ignorando su alta correlación física.
* **Soluciones alcanzadas**: 1) Clasificación: paso de pérdidas MSE a Entropía Cruzada y Focal Loss. 2) Localización: desarrollo de pérdidas Smooth L1 y, posteriormente, pérdidas basadas en IoU (IoU loss, GIoU, DIoU, CIoU) que optimizan de forma unificada la coincidencia de área, la distancia de centros y la relación de aspecto.

##### 5) Technical Evolution of Non-Maximum Suppression
* **Problemas atacados**: Eliminar las detecciones redundantes y superpuestas sobre el mismo objeto físico.
* **Limitaciones de ese entonces**: El NMS tradicional por selección codiciosa (greedy) aplica un umbral estricto que elimina objetos verdaderos adyacentes en zonas muy congestionadas y no suprime falsos positivos sistemáticos.
* **Soluciones alcanzadas**: Evolución en 4 corrientes (Fig. 8): 1) Selección Codiciosa con mejoras: decaimiento suave de confianza (Soft-NMS, Softer-NMS) o umbrales adaptativos (Adaptive-NMS). 2) Agregación de Cajas: fusionar cajas traslapadas mediante clustering (VJ, Overfeat, WBF). 3) NMS por Aprendizaje: redes entrenadas para suprimir duplicados (RelationNet, LearnNMS) y 4) Detectores libres de NMS: correspondencia de un solo objeto a una sola predicción (CenterNet, DETR).

### III. Speed-Up of Detection
Revisión detallada de las tecnologías y metodologías para acelerar el procesamiento de los detectores a nivel de pipeline, arquitectura y cómputo numérico.
* **Problemas atacados**: Viabilizar el despliegue de detectores complejos en plataformas embebidas y dispositivos móviles en tiempo real.
* **Limitaciones de ese entonces**: La alta profundidad de los backbones y la redundancia computacional en múltiples etapas limitaban la adopción en ingeniería práctica.
* **Soluciones alcanzadas**: Clasificación de las técnicas de aceleración en cuatro pilares principales (Fig. 9): computación compartida, cascadas de descarte, optimización numérica e ingeniería de red ligera.

#### A. Feature Map Shared Computation
* **Problemas atacados**: Eliminar la redundancia al extraer características de cientos de RoIs superpuestos.
* **Limitaciones de ese entonces**: R-CNN requería miles de pasadas por la CNN por cada imagen, saturando las capacidades de hardware.
* **Soluciones alcanzadas**: Compartir el mapa de características calculando la CNN sobre la imagen completa una sola vez (SPPNet, Fast R-CNN), acelerando la inferencia cientos de veces.

#### B. Cascaded Detection
* **Problemas atacados**: Procesar de manera eficiente escenas grandes que contienen objetos pequeños y dispersos.
* **Limitaciones de ese entonces**: Evaluar redes densas sobre zonas vacías gigantescas desperdicia recursos.
* **Soluciones alcanzadas**: Uso del enfoque de cascada (VJ detector, face detectors recientes) que descarta rápidamente regiones de fondo simple usando subredes ligeras y reserva los cómputos complejos solo para candidatos difíciles.

#### C. Network Pruning and Quantification
* **Problemas atacados**: Comprimir el peso físico de las redes y reducir los tiempos de ciclo de reloj en CPU/GPU.
* **Limitaciones de ese entonces**: Modelos con millones de parámetros de punto flotante de 32 bits (FP32) son demasiado grandes para chips embebidos.
* **Soluciones alcanzadas**: 1) Poda de Red: eliminar de forma iterativa pesos o canales de convolución poco importantes. 2) Cuantización: convertir parámetros FP32 a formatos de baja precisión (INT8 o variables binarias INT1) para acelerar operaciones a nivel lógico básico.

#### D. Lightweight Network Design
Diseño de bloques de convolución compactos optimizados para alta eficiencia.
* **Problemas atacados**: Diseñar arquitecturas de red ligeras desde su concepción geométrica.
* **Limitaciones de ese entonces**: Las convoluciones bidimensionales tradicionales de gran tamaño (p. ej. $7 \times 7$ o $5 \times 5$) tienen costos de cómputo inaceptables en dispositivos móviles.
* **Soluciones alcanzadas**: Implementación de cuatro estrategias fundamentales de diseño que se detallan a continuación.

##### 1) Factorizing Convolutions
* **Problemas atacados**: Reducir el número de parámetros convolucionales manteniendo el mismo campo receptivo.
* **Limitaciones de ese entonces**: Convoluciones pesadas con alta cantidad de parámetros redundantes.
* **Soluciones alcanzadas**: Factorización espacial: descomponer un filtro de $7 \times 7$ en tres filtros de $3 \times 3$ conectados en serie, reduciendo drásticamente los parámetros manteniendo el mismo campo visual.

##### 2) Group Convolution
* **Problemas atacados**: Dividir el flujo de procesamiento de canales convolucionales.
* **Limitaciones de ese entonces**: Operaciones densas que conectan todos los canales de entrada con todos los canales de salida.
* **Soluciones alcanzadas**: Convolución por grupos: dividir los canales en $m$ grupos independientes para realizar operaciones por separado, reduciendo teóricamente el costo de cómputo a $1/m$.

##### 3) Depth-wise Separable Convolution
* **Problemas atacados**: Minimizar la carga de cómputo dividiendo la convolución espacial de la convolución de canales.
* **Limitaciones de ese entonces**: Convoluciones estándar que mezclan dimensiones espaciales y de canal simultáneamente.
* **Soluciones alcanzadas**: Bloques de convolución separable profunda (implementados en MobileNet): combinación de una convolución por canal (depth-wise) seguido de una convolución de punto $1 \times 1$ (point-wise), bajando el costo computacional hasta en un 90%.

##### 4) Channel Shuffle
* **Problemas atacados**: Superar la falta de comunicación entre canales generada al usar convoluciones por grupo.
* **Limitaciones de ese entonces**: Bloques agrupados que aíslan la información e impiden la mezcla de características semánticas de diferentes canales.
* **Soluciones alcanzadas**: Mecanismo de barajado de canales (Channel Shuffle en ShuffleNet) que mezcla y redistribuye activamente los canales entre grupos para preservar la riqueza semántica de la red.

### IV. State-of-the-Art Detection Methods in the Recent Three Years
Análisis de las tendencias de investigación de vanguardia más recientes hasta 2022.
* **Problemas atacados**: Incorporar transformaciones de atención profunda y optimizar la precisión espacial a nivel industrial.
* **Limitaciones de ese entonces**: Las CNNs sufren para modelar relaciones globales de larga distancia por el tamaño limitado de su núcleo convolucional.
* **Soluciones alcanzadas**: Transición hacia detectores basados en Transformers de visión (como Swin Transformer y variantes de DETR) y la optimización de detectores de un solo disparo comerciales (YOLOv4, YOLOv7) mediante reparametrización estructural en inferencia y asignación dinámica de etiquetas en entrenamiento.

### V. Conclusion and Future Research Directions
Síntesis final del survey y proyección de los retos abiertos en la visión por computadora.
* **Problemas atacados**: Identificar las fronteras científicas de la detección de objetos para los próximos años.
* **Limitaciones de ese entonces**: Falta de pautas sobre cómo integrar modelos de detección con inteligencia artificial generalizada y aprendizaje continuo.
* **Soluciones alcanzadas**: Definición de 7 líneas prioritarias de investigación futura: 1) detección bajo supervisión débil o nula (zero-shot/few-shot), 2) detección de objetos en vídeo en tiempo real, 3) detección multimodal de lenguaje e imagen, 4) modelos de visión unificados (Unified Vision Models), 5) optimización de arquitecturas basadas en Transformers, 6) detección 3D y espacial y 7) aprendizaje continuo sobre flujos de datos sin olvido catastrófico.
