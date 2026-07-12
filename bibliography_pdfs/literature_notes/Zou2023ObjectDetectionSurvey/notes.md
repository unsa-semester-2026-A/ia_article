# Object Detection in 20 Years: A Survey

- **Key**: Zou2023ObjectDetectionSurvey
- **Year**: 2023
- **Venue**: Proceedings of the IEEE

## Resumen
Este artículo presenta una revisión exhaustiva del estado del arte y la evolución técnica de la detección de objetos en un lapso temporal de más de un cuarto de siglo (desde la década de 1990 hasta el año 2022). La historia de la detección de objetos se divide en dos grandes épocas: el periodo de detección tradicional basada en características artesanales (antes de 2014) y el periodo de aprendizaje profundo (después de 2014). Los autores trazan una hoja de ruta técnica que detalla detectores históricos fundamentales (Viola-Jones, HOG, DPM, R-CNN, Faster R-CNN, YOLO, SSD, CornerNet, CenterNet y DETR), analiza las métricas y bases de datos más relevantes (PASCAL VOC, ImageNet, MS COCO, Open Images) y desglosa los bloques constructivos esenciales del detector (multiescala, contexto, minería de negativos, funciones de pérdida y NMS). También se realiza un análisis profundo sobre los métodos de aceleración del procesamiento (poda de redes, cuantificación, convoluciones ligeras y aceleración en hardware) y se identifican las principales tendencias recientes de investigación, tales como detectores basados en transformadores (transformers), detección en 3D, videos y aprendizaje de dominio abierto.

## Secciones y Subsecciones

### I. Introducción
Define la tarea de la detección de objetos enfocado en resolver dos preguntas fundamentales: "¿Qué objetos están dónde?". Establece las dos métricas centrales del rendimiento: precisión (clasificación y localización) y velocidad de inferencia.
* **Problemas atacados**: Variabilidad extrema en el punto de vista, la iluminación, la rotación de los objetos, oclusión, detección de objetos pequeños y la necesidad de procesar en tiempo real.
* **Limitaciones de ese entonces**: Dificultad para novices en la comprensión de la vasta cantidad de técnicas de detección y la desconexión histórica entre los conceptos de la visión tradicional y los modelos profundos contemporáneos.
* **Soluciones alcanzadas**: Elaboración de un survey unificado con un enfoque en la evolución histórica, estructurado en pasado (tradicional), presente (aprendizaje profundo y aceleración) y futuro (direcciones abiertas).

### II. Detección de Objetos en 20 Años
Trazado cronológico del desarrollo metodológico agrupándolo por periodos clave e hitos de arquitectura.

* **Problemas atacados**: Comprensión holística de cómo los avances de hardware y representación guiaron el salto de rendimiento.
* **Limitaciones de ese entonces**: La investigación previa se enfocaba en mejoras aisladas de precisión sin examinar cómo los bloques de construcción individuales (ej. NMS o anclas) evolucionaban colectivamente.
* **Soluciones alcanzadas**: Construcción de mapas de ruta detallados para detectores, datasets y evoluciones de diseño.

#### II.A Una Hoja de Ruta para la Detección de Objetos
Divide los hitos entre detectores tradicionales, de dos etapas y de una etapa.
* **Problemas atacados**: Evolución de la precisión en los conjuntos de datos de referencia (Figura 3).
* **Limitaciones de ese entonces**: Los detectores tradicionales tenían precisión baja y saturada debido al coste de diseñar a mano los descriptores visuales.
* **Soluciones alcanzadas**: Clasificación conceptual de los modelos clave en las últimas dos décadas.

##### Milestones: Traditional Detectors (Hitos: Detectores Tradicionales)
* **Problemas atacados**: Detección en tiempo real con CPU y modelado de la deformación interna de los objetos.
* **Limitaciones de ese entonces**: La capacidad de procesamiento limitada de las computadoras de los años 2000 impedía ejecutar búsquedas densas por ventana deslizante.
* **Soluciones alcanzadas**: 1) Viola-Jones (VJ, 2001) para rostros, implementando la imagen integral, selección Haar mediante Adaboost y cascadas de decisión para descartar fondos rápido. 2) HOG (2005) para peatones, introduciendo el Histograma de Gradientes Orientados normalizado por bloques. 3) DPM (2008), adoptando la filosofía de "divide y vencerás" al modelar objetos como un conjunto de partes interconectadas (star-model y mixture-model), introduciendo bounding box regression y hard negative mining.

##### Milestones: CNN-based Two-stage Detectors (Hitos: Detectores de Dos Etapas Basados en CNN)
* **Problemas atacados**: Superar el cuello de botella de los descriptores artesanales explotando representaciones profundas.
* **Limitaciones de ese entonces**: R-CNN clásico (2014) requería procesar individualmente más de 2000 propuestas de región por imagen mediante CNN, causando extrema lentitud (14s/imagen).
* **Soluciones alcanzadas**: 1) SPPNet (2014), que introduce la capa de Spatial Pyramid Pooling para procesar el mapa de características global una sola vez, logrando aceleraciones de 20x. 2) Fast R-CNN (2015), que integra clasificación y regresión de cajas bajo una pérdida multitarea unificada de entrenamiento extremo a extremo. 3) Faster R-CNN (2015), que incorpora el Region Proposal Network (RPN) para generar candidatos casi gratis. 4) FPN (2017), que introduce la pirámide de características con conexiones laterales para resolver la invariancia a la escala.

##### Milestones: CNN-based One-stage Detectors (Hitos: Detectores de Una Etapa Basados en CNN)
* **Problemas atacados**: Eliminación del paso de propuestas de región para lograr inferencias en tiempo real y dispositivos móviles.
* **Limitaciones de ese entonces**: Los detectores de una sola etapa iniciales (ej. YOLOv1) sufrían de menor precisión de localización y problemas graves al detectar objetos pequeños y agrupados en comparación con redes R-CNN.
* **Soluciones alcanzadas**: 1) YOLO (2015), mapeando las detecciones directamente en una sola cuadrícula de inferencia. 2) SSD (2015), introduciendo la predicción multireferencia sobre mapas multiresolución en diferentes profundidades. 3) RetinaNet (2017), solucionando el desbalance de clases fondo/objeto mediante la pérdida Focal Loss. 4) CornerNet (2018) y CenterNet (2019), proponiendo la detección libre de anclas (anchor-free) tratando cajas como puntos clave. 5) DETR (2020) y Deformable DETR (2021), eliminando la dependencia de anclas y NMS mediante transformadores y predicción de conjuntos.

#### II.B Conjuntos de Datos y Métricas para la Detección de Objetos
Analiza el impacto y tamaño de los datasets históricos (Tabla I).
* **Problemas atacados**: Estandarización y reducción de sesgos en la evaluación de detectores.
* **Limitaciones de ese entonces**: Los primeros datasets eran pequeños o se evaluaban mediante métricas defectuosas a nivel de ventana (como FPPW) que no reflejaban el mAP en imágenes globales reales.
* **Soluciones alcanzadas**: Estandarización del Average Precision (AP) con IoU $\ge$ 0.5 (introducido en PASCAL VOC07) y la métrica promediada en rango de IoU 0.5 a 0.95 de MS COCO para incentivar la localización precisa.

#### II.C Evolución Técnica en la Detección de Objetos
Detalla el progreso tecnológico de los 5 subcomponentes clave del detector.

* **Problemas atacados**: Desajustes en los algoritmos internos de los detectores en el transcurso del tiempo.
* **Limitaciones de ese entonces**: Falta de claridad en cómo el contexto, la minería de negativos y las pérdidas evolucionaban para mitigar los sesgos espaciales.
* **Soluciones alcanzadas**: Análisis históricos de evolución representados en diagramas de hitos (Figuras 5 a 8).

##### Evolución Técnica de la Detección Multiescala
* **Problemas atacados**: Robustez ante objetos con ratios de aspecto y tamaños sumamente divergentes.
* **Limitaciones de ese entonces**: Deslizar ventanas fijas es computacionalmente prohibitivo para múltiples relaciones de aspecto.
* **Soluciones alcanzadas**: Transición desde pirámides de imágenes + ventana deslizable hacia propuestas de objetos bottom-up (Selective Search), regresión directa profunda, detección multiresolución en mapas de características (FPN/SSD), y finalmente predicción libre de anclas basada en esquinas o puntos representativos (CornerNet/Reppoints).

##### Evolución Técnica del Cebado de Contexto (Context Priming)
* **Problemas atacados**: Explotar la información del entorno circundante para guiar la detección.
* **Limitaciones de ese entonces**: Modelar el contexto global con descriptores estadísticos simples (como Gist) perdía resolución espacial local.
* **Soluciones alcanzadas**: Evolución desde el uso de bordes de contorno local (Sinha y Torralba) hacia la integración de contexto global mediante convoluciones dilatadas/deformables de gran receptive field y, recientemente, mecanismos de atención global (Non-Local, Transformers) que calculan relaciones entre todos los píxeles de la imagen.

##### Evolución Técnica de la Minería de Ejemplos Negativos Difíciles (Hard Negative Mining)
* **Problemas atacados**: Manejar el desbalance extremo de muestras clase/fondo (que llega hasta ratios de $10^7:1$).
* **Limitaciones de ese entonces**: En las primeras etapas de las CNNs profundas (2014-2016), el bootstrap se descartó debido a la alta capacidad de cómputo, provocando que redes como YOLO sufrieran con fondos complejos.
* **Soluciones alcanzadas**: Reintroducción de la selección interactiva de muestras (OHEM, RefineDet) y la optimización directa mediante funciones de pérdida que penalizan ejemplos difíciles (Focal Loss).

##### Evolución Técnica de la Función de Pérdida
* **Problemas atacados**: Optimizar de forma diferenciable la clasificación y la localización de cajas.
* **Limitaciones de ese entonces**: Las pérdidas L2 de coordenadas tratan el centro y tamaño del cuadro como variables independientes, lo que no correlaciona directamente con el IoU final de evaluación.
* **Soluciones alcanzadas**: Transición de pérdidas cuadráticas de regresión hacia Smooth L1 y, posteriormente, pérdidas formuladas directamente sobre el IoU (IoU loss, GIoU, DIoU, CIoU) que optimizan simultáneamente el solapamiento, distancia de centros y la relación de aspecto.

##### Evolución Técnica de la Supresión de No Máximos (NMS)
* **Problemas atacados**: Eliminar detecciones duplicadas en objetos adyacentes.
* **Limitaciones de ese entonces**: El NMS codicioso (Greedy) con un umbral estricto elimina detecciones válidas en zonas de oclusión densa.
* **Soluciones alcanzadas**: Evolución desde Greedy NMS simple hacia agrupación de cajas (Bounding Box Aggregation en VJ y Overfeat), variantes de atenuación suave de confianza (Soft-NMS, Adaptive-NMS), redes entrenadas para suprimir (Learning to NMS) y detectores puramente libres de NMS mediante emparejamiento uno a uno (DETR, CenterNet).

### III. Aceleración de la Detección
Clasifica los métodos de aceleración en tres niveles: nivel de pipeline de detección, nivel de columna vertebral (backbone) y nivel numérico.

* **Problemas atacados**: Alta latencia e ineficiencia computacional en dispositivos con recursos limitados.
* **Limitaciones de ese entonces**: Los detectores precisos no podían ejecutarse en tiempo real en sistemas embebidos de bajo consumo.
* **Soluciones alcanzadas**: 1) Compartir mapas de características (cómputo unificado). 2) Detección en cascada para filtrar fondos rápidamente. 3) Poda de pesos innecesarios en la CNN (network pruning) y cuantificación de variables a valores binarios. 4) Diseño de capas convolucionales optimizadas: convolución agrupada (Group Conv), convolución separable en profundidad (Depth-wise Separable Conv de MobileNet) y compresión de cuello de botella (Bottleneck). 5) Aceleración matemática de bajo nivel usando imágenes integrales para HOG, convolución en dominio de frecuencia (FFT/IFFT) y cuantización vectorial.

### IV. Avances Recientes en la Detección de Objetos
Analiza los progresos técnicos en los últimos años del periodo de deep learning.

* **Problemas atacados**: Consolidar y fusionar aproximaciones dispares de aprendizaje para mejorar la robustez geométrica y la supervisión.
* **Limitaciones de ese entonces**: La detección clásica se limitaba a supervisión fuerte de cajas horizontales sin adaptación de dominio.
* **Soluciones alcanzadas**: Agrupación de las tendencias en 8 categorías de frontera.

#### IV.A Más allá de la Detección por Ventana Deslizante
* **Problemas atacados**: Superar el paradigma de cuadrícula y ventana.
* **Limitaciones de ese entonces**: La regresión rígida de cajas con anclas introduce exceso de hiperparámetros manuales de inicialización.
* **Soluciones alcanzadas**: Modelar objetos como puntos clave individuales (CenterNet) o conjuntos libres de NMS basados en transformadores (DETR).

#### IV.B Detección Robusta ante Cambios de Rotación y Escala
* **Problemas atacados**: Variabilidad de la orientación y escala de los objetos.
* **Limitaciones de ese entonces**: La pooling convolucional tradicional en coordenadas cartesianas no es invariante a la rotación.
* **Soluciones alcanzadas**: 1) En rotación: aumento de datos masivo, pérdidas invariantes rotacionales, pooling en coordenadas polares. 2) En escala: técnicas de entrenamiento adaptativo multiescala (SNIP, SNIPER) que filtran gradientes fuera del rango óptimo de escala, y reescalado de imágenes basado en la distribución de escala predicha.

#### IV.C Detección con Mejores Extractores de Características (Backbones)
* **Problemas atacados**: Capturar relaciones semánticas de largo alcance.
* **Limitaciones de ese entonces**: El campo receptivo local de las CNNs limita la comprensión contextual global de la escena.
* **Soluciones alcanzadas**: Adopción de arquitecturas Transformer (como Swin Transformer) que superan ampliamente a las CNNs clásicas en la curva precisión/velocidad de MS COCO.

#### IV.D Mejoras en la Localización
* **Problemas atacados**: Refinar bordes y reportar incertidumbre de localización.
* **Limitaciones de ese entonces**: El NMS tradicional falla cuando no dispone de una medida de confianza sobre la precisión de la caja predicha.
* **Soluciones alcanzadas**: Refinamiento iterativo de cajas y el modelado probabilístico de la regresión de coordenadas prediciendo la distribución de probabilidad espacial de las esquinas.

#### IV.E Aprendizaje con Pérdida de Segmentación
* **Problemas atacados**: Guía de entrenamiento mediante pérdidas secundarias de segmentación (multi-task learning).
* **Limitaciones de ese entonces**: La anotación a nivel de píxel es costosa y ralentiza la inferencia si la rama semántica se mantiene activa en test.
* **Soluciones alcanzadas**: Entrenamiento conjunto con una rama de segmentación que se descarta durante la inferencia, guiando las características del backbone sin añadir velocidad de cómputo en test.

#### IV.F Entrenamiento Adversarial
* **Problemas atacados**: Detección de objetos ocluidos y extremadamente pequeños.
* **Limitaciones de ese entonces**: Generar imágenes de súper-resolución en espacio de píxeles es lento y costoso.
* **Soluciones alcanzadas**: Redes GAN que operan directamente sobre el espacio de características, super-resolviendo descriptores de objetos pequeños o inyectando máscaras de oclusión para forzar la robustez del detector.

#### IV.G Detección de Objetos Débilmente Supervisada (WSOD)
* **Problemas atacados**: Entrenar detectores sin disponer de cajas delimitadoras anotadas.
* **Limitaciones de ese entonces**: Detección imprecisa de límites de objetos al entrenar solo con etiquetas de nivel de imagen.
* **Soluciones alcanzadas**: Uso de Aprendizaje por Instancias Múltiples (MIL) considerando candidatos de imagen como sacos con etiquetas, y el mapeo de activación de clases (CAM) para localizar regiones de interés intrínsecas a la clasificación profunda.

#### IV.H Detección con Adaptación de Dominio
* **Problemas atacados**: Resolver la detección sobre datos i.i.d. y reducir el dominio shift.
* **Limitaciones de ese entonces**: Los detectores sufren degradación de mAP masiva cuando se evalúan en condiciones climáticas o entornos geográficos diferentes al entrenamiento.
* **Soluciones alcanzadas**: Uso de regularizadores de características y entrenamiento adversarial a nivel de imagen, clase y objeto, así como traducción de imágenes mediante CycleGAN.

### V. Conclusión y Direcciones Futuras
Identifica los 7 campos de vanguardia que definirán el futuro de la detección de objetos.
* **Lightweight object detection**: Diseñar detectores ultra-ligeros capaces de procesar imágenes de alta resolución en dispositivos de bajo consumo para ciudades inteligentes y vehículos autónomos.
* **End-to-End object detection**: Perfeccionar modelos que eliminen por completo las heurísticas de NMS sin perder precisión ni velocidad.
* **Small object detection**: Detección de objetos pequeños en grandes entornos, crucial en cartografía por satélite y rescate.
* **3D object detection**: Estimación exacta de la localización y pose en el espacio tridimensional combinando RGB con nubes de puntos Lidar.
* **Detection in videos**: Explotar la correlación espacio-temporal continua entre fotogramas de vídeo para mejorar la robustez a bajo coste computacional.
* **Cross-modality detection**: Fusión de sensores que integran audio, vídeo, texto y mapas térmicos para emular la percepción humana.
* **Towards open-world detection**: Detección en un mundo abierto donde los algoritmos identifiquen instancias de categorías desconocidas y actualicen incrementalmente sus parámetros sin sufrir olvido catastrófico (catastrophic forgetting).
