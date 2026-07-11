# Oriented object detection in optical remote sensing images using deep learning: a survey

- **Key**: Wang2025OBBSurvey
- **Year**: 2025
- **Venue**: Artificial Intelligence Review

## Resumen
Este artículo presenta una revisión exhaustiva y estructurada de los avances recientes en la detección de objetos orientados (Oriented Object Detection) en imágenes de percepción remota (Remote Sensing, RS) óptica a través de técnicas de aprendizaje profundo. A diferencia de la detección de objetos horizontal convencional (HBB), la detección orientada utiliza cajas orientadas (Oriented Bounding Boxes, OBB) para ajustar de manera precisa objetos con orientaciones arbitrarias y grandes relaciones de aspecto (como barcos, puentes o vehículos desde vista aérea). El survey establece una taxonomía clara dividida en marcos de detección, técnicas de regresión de OBB, enfoques de representación de características, y soluciones a problemas comunes en imágenes satelitales (fondos complejos, grandes variaciones de escala y escasez de anotaciones). Adicionalmente, resume los conjuntos de datos de referencia (como DOTA, DIOR-R y FAIR1M), analiza el desempeño de modelos estado del arte (incluyendo CNNs y Vision Transformers), y discute direcciones futuras como el despliegue ligero y modelos multimodales a gran escala.

## Secciones y Subsecciones

### 1 Introduction
Introduce el papel fundamental de la detección de objetos en imágenes satelitales y de vehículos aéreos no tripulados (UAVs) para aplicaciones civiles e industriales. Explica por qué las perspectivas de adquisición de ojo de pájaro (BEV) requieren OBBs en lugar de HBBs (para evitar el solapamiento extremo en áreas densas y reducir el ruido de fondo).

#### 1.1 Comparisons with related surveys
Compara este trabajo con surveys previos sobre detección general y de percepción remota, destacando que este es el primer survey enfocado de manera profunda y sistemática exclusivamente en la detección de objetos rotados/orientados.

#### 1.2 Scope
Establece los criterios de selección de artículos (centrados en publicaciones influyentes de primer nivel en el periodo de explosión del área entre 2019 y 2024).

#### 1.3 Contributions
Sintetiza las cuatro contribuciones principales: revisión de la evolución técnica, taxonomía exhaustiva de métodos, comparación de SOTA en benchmarks y discusión de problemas abiertos.

### 2 From horizontal object detection to oriented object detection
Traza el camino histórico desde los métodos tradicionales basados en características hechas a mano (como SIFT o HOG) y clasificadores clásicos (SVM) hasta las redes convolucionales profundas (CNN) y Vision Transformers (ViT).

#### Desafíos Clave de OBB frente a HBB:
- **Feature misalignment (desalineación de características):** Desajuste entre las características extraídas mediante convoluciones alineadas con los ejes de la imagen y la orientación arbitraria de los objetos reales.
- **Spatial misalignment (desalineación espacial):** Ineficiencia de las cajas de anclaje (anchors) horizontales para ajustarse a objetos rotados con grandes relaciones de aspecto.
- **OBB regression problems (problemas de regresión):** Dificultades como la periodicidad de los ángulos (PoA), que causa discontinuidades en los límites angulares, y el problema del orden de los vértices en representaciones cuadrilaterales.

### 3 Detection frameworks
Clasifica los detectores en cuatro paradigmas principales:

#### 3.1 Two-stage detectors
(ej., Rotated Faster R-CNN, RoI Transformer, Oriented RCNN). Emplean una propuesta gruesa en la primera etapa (RPN) y refinamientos en la segunda. El RoI Transformer y el Oriented RCNN representan hitos para generar propuestas rotadas eficientes y alinear características espacialmente a través de RRoI operators.

#### 3.2 One-stage detectors
(ej., R3Det, S2A-Net, CFL). Clasifican y regresan en un solo paso. Para alinear características y solventar el feature misalignment usan módulos de refinamiento (FRM) o convoluciones de alineación (AlignConv) guiadas por las anclas estimadas.

#### 3.3 Anchor-free detectors
Evitan el uso de anclas fijas para solucionar el spatial misalignment. Se dividen en:
- **Keypoint-based (basados en puntos clave):** Ubican puntos como esquinas o centros para derivar la OBB (ej., O2-DNet, CFA, Oriented RepPoints).
- **Center-based (basados en centros):** Generan mapas de calor probabilísticos del centro y regresan parámetros rotados (ej., CHPDet, GGHL, DRDet, TS-Conv).

#### 3.4 DETR-based detectors
Adaptaciones del transformador de detección (ej., O2-DETR, AO2-DETR, ARS-DETR) que tratan la detección como una predicción de conjuntos, eliminando componentes heurísticos como NMS, aunque sufren de convergencia lenta en la optimización angular.

### 4 OBB regression technologies
Analiza cómo optimizar el backpropagation ante las inconsistencias de pérdida y ángulo.

#### 4.1 Regression loss
Aborda la inconsistencia métrica-pérdida (donde minimizar L1 no asegura un RIoU alto). Presenta pérdidas diferenciables aproximadas al RIoU (como PIoU, GCL) y modelos basados en distancia de distribuciones gaussianas (ej., GWD, KLD, KFIoU) que transforman la OBB en una gaussiana 2D para calcular distancias (como Wasserstein o Kullback-Leibler) y eludir la discontinuidad angular.

#### 4.2 OBB representation
Evalúa diferentes formas de representar la caja: cuadriláteros ordenados, métodos basados en puntos clave sin orden, offsets respecto a los puntos medios (Midpoint Offset) y representaciones polares o basadas en puntos de cabeza (head points) para indicar direcciones unívocas en un rango de $2\pi$.

### 5 Feature representation approaches
Se enfoca en redes diseñadas para mejorar la calidad intrínseca de los mapas de características ante rotaciones.

#### 5.1 Rotation-invariant feature representations
Redes como ReDet y ARC que extraen características equivariantes o invariantes a la rotación mediante convoluciones equivariantes de grupo (G-CNNs) o rotación dinámica de los núcleos de convolución en función de la orientación del objeto.

#### 5.2 Advanced feature representations
La aplicación de Vision Transformers a gran escala para percepción remota. Destaca el mecanismo de atención con ventanas de tamaño variable rotadas (RVSA) y el método STD (divide-and-conquer) para refinar progresivamente las características según máscaras de activación de OBB desacopladas.

### 6 Solutions to common issues
Resuelve problemas propios del entorno físico y operacional de RS.

#### 6.1 Complex backgrounds
Uso de mecanismos de atención espacial/canal guiados y eliminación de ruido a nivel de instancia (como SCRDet++ e InLD) para suprimir texturas del suelo que imitan a los objetos.

#### 6.2 Scale variations
Soluciones a nivel de red (pirámides de características jerárquicas como FPN y kernels múltiples paralelos) y asignadores de muestras adaptativos (como el assigner coarse-to-fine de Xu et al. o priorización dinámica posterior).

#### 6.3 Large aspect ratios
Adaptaciones en pérdidas (ej., pérdidas angulares ponderadas por la relación de aspecto) y métricas modificadas (como la longitud independiente IoU, LiIoU) para asegurar el aprendizaje en objetos alargados.

#### 6.4 Lack of orientation-annotated samples
Técnicas semi-supervisadas (estructuras Teacher-Student con pseudo-etiquetado rotado consistente) y débilmente supervisadas (ej., H2RBox y Point2RBox) para entrenar detectores rotados utilizando únicamente anotaciones horizontales (HBB) o de puntos clave.

### 7 Evaluation protocols and datasets
Describe las métricas estándar de precisión basadas en RIoU (Rotated IoU), precisión, recall y Average Precision (AP y mAP), y realiza un censo comparativo de los principales datasets como DOTA (V1.0, V1.5, V2.0), DIOR-R, ShipRSImageNet, DroneVehicle y FAIR1M.

### 8 State-of-the-art methods
Compara los resultados experimentales en DOTA-V1.0. Demuestra que las arquitecturas basadas en dos etapas (Oriented RCNN) y la integración de Vision Transformers (ViT con RVSA o STD) logran el mejor desempeño en mAP, seguidos de cerca por los detectores de una etapa con refinamiento, mientras que los modelos basados en DETR continúan rezagados debido a las dificultades del paradigma de queries para cubrir ángulos finos.

### 9 Conclusions and future directions
Identifica tres limitaciones críticas: baja eficiencia de detección por la excesiva complejidad, desbalance extremo de clases en conjuntos de datos y la dependencia de imágenes monomodales. Delinea direcciones futuras:
- **Lightweight methods:** Compresión (poda y cuantización) y destilación de conocimiento.
- **Mission-specific datasets:** Colección de datos multimodales y a gran escala.
- **Multimodal large models:** Integración de VLM (Vision-Language Models) capaces de razonar con inputs satelitales, telemetría GPS/IMU y lenguaje natural de manera unificada.
