# Oriented object detection in optical remote sensing images using deep learning: a survey

- **Key**: Wang2025OBBSurvey
- **Year**: 2025
- **Venue**: Artificial Intelligence Review

## Resumen

Este artículo presenta una revisión exhaustiva y estructurada de los avances recientes en la detección de objetos orientados (Oriented Object Detection, OOD) en imágenes de percepción remota (Remote Sensing, RS) óptica a través de técnicas de aprendizaje profundo. A diferencia de la detección de objetos horizontal convencional (HBB), la detección orientada utiliza cajas orientadas (Oriented Bounding Boxes, OBB) para ajustar de manera precisa objetos con orientaciones arbitrarias y grandes relaciones de aspecto (como barcos, puentes o vehículos desde vista aérea). El survey establece una taxonomía clara dividida en marcos de detección, técnicas de regresión de OBB, enfoques de representación de características, y soluciones a problemas comunes en imágenes satelitales (fondos complejos, grandes variaciones de escala y escasez de anotaciones). Adicionalmente, resume los conjuntos de datos de referencia (como DOTA, DIOR-R y FAIR1M), analiza el desempeño de modelos estado del arte (incluyendo CNNs y Vision Transformers), y discute direcciones futuras como el despliegue ligero y modelos multimodales a gran escala.

## Secciones y Subsecciones

### 1 Introduction
Introduce el papel fundamental de la detección de objetos en imágenes satelitales y de vehículos aéreos no tripulados (UAVs) para aplicaciones civiles e industriales. Explica por qué las perspectivas de adquisición de ojo de pájaro (BEV) requieren OBBs en lugar de HBBs (para evitar el solapamiento extremo en áreas densas y reducir el ruido de fondo).
* **Problemas atacados**: Insuficiencia de los métodos horizontales (HBB) para detectar objetos en imágenes de teledetección (RS), donde los objetos tienen orientaciones arbitrarias y alta densidad.
* **Limitaciones de ese entonces**: Los detectores horizontales convencionales sufren de solapamiento extremo (IoU alta entre HBBs adyacentes) en zonas congestionadas, provocando que la técnica NMS elimine detecciones válidas.
* **Soluciones alcanzadas**: Introducción del concepto de OBB para ajustar estrechamente objetos orientados, proporcionando localización precisa e información angular.

#### 1.1 Comparisons with related surveys
Compara este trabajo con surveys previos sobre detección general y de percepción remota.
* **Problemas atacados**: Falta de una revisión sistemática, profunda y actualizada dedicada exclusivamente a la detección de objetos rotados en percepción remota.
* **Limitaciones de ese entonces**: Revisiones previas cubrían OOD de forma muy superficial o se limitaban a resumir métodos históricos sin clasificaciones detalladas o análisis de trade-offs.
* **Soluciones alcanzadas**: Se proporciona una taxonomía comprensiva con análisis profundo de los pros y contras de cada método, analizando el rendimiento de modelos CNN y ViT recientes.

#### 1.2 Scope
Establece los criterios de selección de artículos para delimitar la literatura analizada.
* **Problemas atacados**: Filtrar y estructurar el volumen masivo de publicaciones sobre OOD aparecidas entre 2019 y 2024.
* **Limitaciones de ese entonces**: El crecimiento exponencial de la literatura hace imposible e ineficaz revisar todas las publicaciones sin un criterio de impacto y rigor.
* **Soluciones alcanzadas**: Delimitación del alcance a artículos de OOD monovista e influyentes publicados en conferencias y revistas de primer nivel mundial.

#### 1.3 Contributions
Sintetiza las cuatro contribuciones principales del survey.
* **Problemas atacados**: La fragmentación del conocimiento técnico y la falta de un marco comparativo coherente en OOD.
* **Limitaciones de ese entonces**: Los investigadores carecían de una guía integral que trazara la evolución de HBB a OBB, clasificaciones rigurosas de regresiones OBB, y direcciones de investigación abiertas.
* **Soluciones alcanzadas**: Propuesta de una taxonomía de 4 partes (frameworks, regresiones, características y problemas comunes) con comparativas detalladas y una hoja de ruta para trabajos futuros.

### 2 From horizontal object detection to oriented object detection
Traza el camino histórico desde los métodos tradicionales basados en características hechas a mano hasta las redes convolucionales profundas y Vision Transformers.
* **Problemas atacados**: Explicar la necesidad de la transición tecnológica de HBB a OBB y clasificar los retos fundamentales que surgen en esta transición.
* **Limitaciones de ese entonces**: Los descriptores tradicionales (SIFT, HOG, SVM) tenían baja capacidad de representación. Los adaptadores directos HBB-to-OBB fallaban por desalineaciones espaciales y de características.
* **Soluciones alcanzadas**: Identificación y conceptualización de tres desafíos principales específicos de OBB: feature misalignment, spatial misalignment, y OBB regression problems.

#### 2.1 Feature misalignment
Describe el desajuste entre características extraídas y la orientación del objeto.
* **Problemas atacados**: Desajuste entre las características extraídas mediante convoluciones estándar alineadas con los ejes y la orientación real del objeto rotado.
* **Limitaciones de ese entonces**: Las convoluciones axis-aligned extraen información de fondo ruidoso en lugar del objeto orientado, degradando la clasificación y regresión.
* **Soluciones alcanzadas**: Desarrollo de operadores RRoI y convoluciones de alineamiento para extraer representaciones espacialmente consistentes del objeto rotado.

#### 2.2 Spatial misalignment
Analiza la ineficiencia de anclajes horizontales para objetos rotados.
* **Problemas atacados**: Limitación de cajas de anclaje (anchors) horizontales fijas para solaparse adecuadamente con objetos orientados de gran relación de aspecto.
* **Limitaciones de ese entonces**: La asignación de muestras positivas basada en IoU estándar falla ante objetos alargados e inclinados, proporcionando muestras insuficientes durante el entrenamiento.
* **Soluciones alcanzadas**: Introducción de anclas rotadas y estrategias de asignación dinámica que no dependen únicamente de la geometría estática horizontal.

#### 2.3 OBB regression problems
Detalla los problemas matemáticos del backpropagation angular y de vértices.
* **Problemas atacados**: Inestabilidad del entrenamiento debido a la periodicidad del ángulo (PoA) y el ordenamiento inconsistente de vértices.
* **Limitaciones de ese entonces**: Pequeñas desviaciones angulares cerca del límite de rango (ej., $\pi/2$) provocaban saltos gigantescos de pérdida (boundary discontinuity), confundiendo al optimizador.
* **Soluciones alcanzadas**: Desarrollo de pérdidas basadas en distribuciones (GWD, KLD) y representaciones libres de ángulo (midpoint offset, quadrant points).

### 3 Detection frameworks
Clasifica los detectores en cuatro paradigmas principales.
* **Problemas atacados**: Sistematizar las diversas arquitecturas de red diseñadas para OOD en el estado del arte.
* **Limitaciones de ese entonces**: Falta de una clasificación estructural que permita a los investigadores seleccionar el framework óptimo según velocidad y precisión.
* **Soluciones alcanzadas**: Categorización en detectores de dos etapas, de una etapa, anchor-free y basados en DETR, analizando cómo cada uno aborda las desalineaciones.

#### 3.1 Two-stage detectors
Describe modelos como Rotated Faster R-CNN, RoI Transformer y Oriented RCNN.
* **Problemas atacados**: Generación de propuestas orientadas de alta calidad en la primera etapa para refinar en la segunda.
* **Limitaciones de ese entonces**: El uso de anclas rotadas densas (RRPN) provoca una explosión computacional y de memoria insostenible.
* **Soluciones alcanzadas**: Desarrollo de módulos ligeros como el RoI learner (RoI Transformer) o midpoint offset (Oriented RCNN) que generan propuestas orientadas eficientemente.

#### 3.2 One-stage detectors
Analiza modelos como R3Det y S2A-Net.
* **Problemas atacados**: Clasificación y regresión angular directa en un solo paso para alta velocidad.
* **Limitaciones de ese entonces**: Ausencia de una etapa de propuesta que provoca desalineación severa de características.
* **Soluciones alcanzadas**: Uso de módulos de refinamiento (FRM) o convoluciones de alineación guiadas (AlignConv) para alinear características en vuelo.

#### 3.3 Anchor-free detectors
Examina detectores que evitan el uso de anclas predefinidas.
* **Problemas atacados**: Evitar los hiperparámetros y la complejidad espacial asociados a las anclas fijas.
* **Limitaciones de ese entonces**: Los detectores con anclas fallan en objetos densos de diferentes escalas e inclinaciones.
* **Soluciones alcanzadas**: Enfoques basados en puntos clave (Oriented RepPoints) y basados en mapas de calor del centro (GGHL, TS-Conv) que predicen directamente las OBBs.

#### 3.4 DETR-based detectors
Modelos adaptados del transformador de detección (O2-DETR, AO2-DETR, ARS-DETR).
* **Problemas atacados**: Eliminar componentes heurísticos complejos diseñados a mano como NMS y la generación de anclas.
* **Limitaciones de ese entonces**: Convergencia de entrenamiento extremadamente lenta y alto coste computacional w.r.t. modelos convolucionales.
* **Soluciones alcanzadas**: Planteamiento de la detección como predicción de conjuntos usando queries de atención orientada y atenciones deformables adaptadas.

#### 3.5 Discussion
Discusión crítica de los frameworks de detección.
* **Problemas atacados**: Analizar el panorama comparativo global y las brechas persistentes en los frameworks.
* **Limitaciones de ese entonces**: Los modelos de dos etapas dominaban en precisión pero eran lentos; los DETR rotados seguían inmaduros para la optimización fina de ángulos.
* **Soluciones alcanzadas**: Conclusión de que la combinación de etapas anchor-free de propuesta seguidas de alineamiento convolucional representa el equilibrio óptimo actual.

### 4 OBB regression technologies
Analiza la optimización del backpropagation mediante pérdidas y representaciones rediseñadas.
* **Problemas atacados**: El error sistemático introducido por la discontinuidad de frontera angular y la inconsistencia métrica-pérdida.
* **Limitaciones de ese entonces**: Minimizar la pérdida L1 estándar no garantiza una IoU rotada alta, y el ángulo $\theta$ es discontinuo.
* **Soluciones alcanzadas**: Replanteamiento de la regresión angular mediante métricas probabilísticas y representaciones geométricas alternativas.

#### 4.1 Regression loss
Revisa pérdidas como PIoU, GWD y KLD.
* **Problemas atacados**: La no diferenciabilidad de la métrica RIoU que impide optimizarla directamente en el backpropagation.
* **Limitaciones de ese entonces**: Las pérdidas clásicas (smooth L1) sufren del problema PoA y no capturan la verdadera superposición rotada.
* **Soluciones alcanzadas**: Transformación de la OBB en una distribución gaussiana bidimensional para calcular distancias diferenciables (Wasserstein en GWD, Kullback-Leibler en KLD).

#### 4.2 OBB representation
Estudia diferentes formas de representar la caja inclinada.
* **Problemas atacados**: Evitar la regresión explícita del ángulo $\theta$ para eludir las singularidades matemáticas.
* **Limitaciones de ese entonces**: La representación basada en vértices ordenados sufre de inconsistencia en las secuencias durante el entrenamiento.
* **Soluciones alcanzadas**: Uso de coordenadas polares, representaciones basadas en puntos medios u offsets de cuadrantes, y el uso de head points para dirección unívoca en $2\pi$.

#### 4.3 Discussion
Resumen del panorama de regresión.
* **Problemas atacados**: Comparar y contrastar la robustez de las pérdidas y representaciones propuestas.
* **Limitaciones de ese entonces**: Pocas representaciones OBB novedosas integraban soluciones al problema de inconsistencia métrica-pérdida de forma directa.
* **Soluciones alcanzadas**: Las pérdidas gaussianas (GWD/KLD) emergen como la solución matemática más elegante y generalizada para evitar angularidades discontinuas.

### 5 Feature representation approaches
Se enfoca en redes diseñadas para mejorar la calidad intrínseca de los mapas de características ante rotaciones.
* **Problemas atacados**: La sensibilidad al ángulo en los extractores de características convencionales.
* **Limitaciones de ese entonces**: Las convoluciones tradicionales no son equivariantes a rotación, requiriendo augmentaciones densas ineficientes.
* **Soluciones alcanzadas**: Diseño de arquitecturas de red con invariancia/equivarianza rotacional matemática interna y adopción de auto-atenciones dinámicas.

#### 5.1 Rotation-invariant feature representations
Revisa convoluciones equivariantes de grupo (G-CNNs) y redes como ReDet.
* **Problemas atacados**: Extracción de características robustas y estables que no varíen con el ángulo de orientación del objeto.
* **Limitaciones de ese entonces**: Las rotaciones aleatorias en entrenamiento solo aproximan la invariancia, aumentando el sobreajuste.
* **Soluciones alcanzadas**: Redes G-CNN que propagan características en múltiples canales de orientación y operadores de alineamiento (ORAlign) en el neck.

#### 5.2 Advanced feature representations
La aplicación de Vision Transformers (ViTs) y ConvNeXt.
* **Problemas atacados**: Capturar el contexto de largo alcance y relaciones semánticas de los objetos satelitales.
* **Limitaciones de ese entonces**: Los ViT estándar procesan ventanas cuadradas estáticas que cortan objetos alargados rotados e incorporan mucho ruido de fondo.
* **Soluciones alcanzadas**: Desarrollo de RVSA (atención con ventana rotada de tamaño variable) y redes de refinamiento progresivo (STD) basadas en máscaras de activación de OBB.

#### 5.3 Discussion
Discusión sobre la representación de características.
* **Problemas atacados**: Identificar la brecha entre los extractores convolucionales equivariantes y los potentes modelos ViT.
* **Limitaciones de ese entonces**: Los extractores equivariantes clásicos (G-CNNs) se basaban en CNNs antiguas y lentas, quedando rezagados frente a los transformers modernos.
* **Soluciones alcanzadas**: Se promueve como investigación crítica la convergencia de invariancia rotacional matemática con transformers de atención deformables.

### 6 Solutions to common issues
Resuelve problemas propios del entorno físico y operacional de las imágenes satelitales.
* **Problemas atacados**: Mitigar interferencias operacionales críticas (fondos complejos, variaciones de escala, escasez de datos).
* **Limitaciones de ese entonces**: Los detectores teóricos fallan en campo debido a la inmensidad y variabilidad de la superficie terrestre.
* **Soluciones alcanzadas**: Desarrollo de atenciones de foreground guiadas, parches dinámicos, adaptaciones por aspect-ratio y aprendizaje débilmente supervisado.

#### 6.1 Complex backgrounds
Uso de mecanismos de atención y eliminación de ruido (SCRDet++, InLD).
* **Problemas atacados**: La alta tasa de falsos positivos causada por texturas del suelo que imitan visualmente a los objetos (e.g. campos, tejados).
* **Limitaciones de ese entonces**: Los módulos de atención convencionales carecían de supervisión directa, activándose en zonas de ruido.
* **Soluciones alcanzadas**: Incorporación de máscaras semánticas supervisadas por la OBB real para apagar activaciones de fondo en canales específicos.

#### 6.2 Scale variations
Soluciones a nivel de red (FPNs híbridas, kernels paralelos) y a nivel de datos (multi-scale).
* **Problemas atacados**: Variaciones de escala extremas (UAV vs satélite) que superan el campo receptivo fijo del modelo.
* **Limitaciones de ese entonces**: El entrenamiento multi-escala clásico degrada severamente el tiempo de procesamiento.
* **Soluciones alcanzadas**: Desarrollo de asignadores dynamically-mined (ATSS rotado, priorización posterior) y convoluciones de kernel múltiple paralelo (PKINet).

#### 6.3 Large aspect ratios
Adaptaciones en pérdidas (LiIoU) y atenciones.
* **Problemas atacados**: Inestabilidad y pérdida de cajas en objetos extremadamente alargados (barcos, puertos, puentes).
* **Limitaciones de ese entonces**: Cajas con pequeñas desviaciones angulares daban IoU cero, eliminando al objeto del entrenamiento.
* **Soluciones alcanzadas**: Adopción de la métrica LiIoU que intercepta cajas, e incorporación de pesos angulares dependientes de la relación de aspecto.

#### 6.4 Lack of orientation-annotated samples
Técnicas semi-supervisadas y débilmente supervisadas (H2RBox, Point2RBox).
* **Problemas atacados**: El coste prohibitivo de etiquetar manualmente orientaciones y cajas rotadas en grandes volúmenes de datos.
* **Limitaciones de ese entonces**: Los datasets HBB existentes no podían usarse directamente para entrenar detectores rotados OBB de alto nivel.
* **Soluciones alcanzadas**: Enfoques de consistencia rotacional (H2RBox) y aprendizaje autosupervisado desde anotaciones de un solo punto (Point2RBox).

#### 6.5 Discussion
Discusión de los problemas comunes de RS.
* **Problemas atacados**: Evaluar la brecha de madurez de las soluciones a problemas comunes.
* **Limitaciones de ese entonces**: La precisión de los modelos débil/semi-supervisados seguía estando muy por detrás de las versiones totalmente supervisadas.
* **Soluciones alcanzadas**: Propuesta de frameworks híbridos weakly-semi-supervised y algoritmos de enfoque (focus-and-detect) para acelerar inferencias en parches vacíos.

### 7 Evaluation protocols and datasets

#### 7.1 Evaluation protocols
Define las métricas de precisión basadas en RIoU, precisión, recall, AP y mAP.
* **Problemas atacados**: Estructurar la metodología de evaluación cuantitativa justa en OOD.
* **Limitaciones de ese entonces**: Evaluar cajas orientadas con IoU horizontal estándar enmascara errores de orientación graves.
* **Soluciones alcanzadas**: Estandarización de las métricas sobre RIoU rotada y desgloses de precisión a diferentes umbrales (AP50, AP75).

#### 7.2 Datasets
Censo comparativo de los principales datasets como DOTA (V1.0, V1.5, V2.0), DIOR-R, ShipRSImageNet, DroneVehicle y FAIR1M.
* **Problemas atacados**: Seleccionar y caracterizar las bases de datos viales y aéreas de referencia.
* **Limitaciones de ese entonces**: Los datasets tempranos (3K vehicle, UCAS-AOD) tenían muy pocas imágenes y variación escasa de escenas.
* **Soluciones alcanzadas**: Compilación detallada de las características de 14 conjuntos de datos, destacando los benchmarks masivos multiclase modernos (DOTA, FAIR1M).

#### 7.3 Discussion
Discusión de datasets y protocolos.
* **Problemas atacados**: Analizar el impacto de la evolución de datos en los detectores.
* **Limitaciones de ese entonces**: Muchos detectores logran saturar los datasets simples, ocultando debilidades ante oclusiones y desbalances del mundo real.
* **Soluciones alcanzadas**: Orientación de la investigación hacia conjuntos masivos con orientaciones libres oblicuas, anotaciones de alta resolución y cross-modalidad (UAV/infrarrojos).

### 8 State-of-the-art methods

#### 8.1 Comparison on DOTA
Analiza los resultados de modelos de punta en el benchmark de referencia DOTA.
* **Problemas atacados**: Evaluar comparativamente la precisión mAP del estado del arte bajo condiciones uniformes.
* **Limitaciones de ese entonces**: Las diferentes librerías y trucos de entrenamiento dificultaban atribuir las mejoras a los componentes principales.
* **Soluciones alcanzadas**: Tabla exhaustiva comparativa en DOTA-v1.0 evidenciando la superioridad de Oriented RCNN y ViT-RVSA/STD.

#### 8.2 Comparison on other datasets
Analiza rendimientos en conjuntos secundarios como HRSC2016 o DIOR-R.
* **Problemas atacados**: Evaluar la generalización de los detectores en dominios específicos (e.g. barcos o infraestructuras).
* **Limitaciones de ese entonces**: Un modelo que sobresale en DOTA no necesariamente generaliza bien en vistas de alta resolución con barcos alargados.
* **Soluciones alcanzadas**: Comparativa unificada demostrando que modelos basados en atenciones de transformers logran mAP superiores al 98% en HRSC2016.

#### 8.3 Discussion
Discusión de los resultados experimentales.
* **Problemas atacados**: Sintetizar los hallazgos empíricos y definir qué paradigmas lideran el rendimiento.
* **Limitaciones de ese entonces**: Los detectores basados en DETR rotados continuaban rezagados en AP50 general frente a detectores de una y dos etapas convolucionales.
* **Soluciones alcanzadas**: Identificación de que la regresión basada en mapas de calor y atenciones de transformers de ventana variable lidera actualmente en precisión y robustez.

### 9 Conclusions and future directions

#### 9.1 Conclusions
Resume las conclusiones principales de la revisión.
* **Problemas atacados**: Consolidar el estado de madurez de la detección de objetos orientados.
* **Limitaciones de ese entonces**: La brecha de rendimiento persistente en objetos pequeños inclinados y la ineficiencia temporal.
* **Soluciones alcanzadas**: Síntesis que sitúa a OOD como un campo clave y consolidado en percepción remota, motivando la integración de nuevas tecnologías.

#### 9.2 Future directions
Delinea la hoja de ruta investigativa.
* **Problemas atacados**: Definir las brechas de investigación más urgentes y proponer líneas de trabajo concretas.
* **Limitaciones de ese entonces**: Dependencia extrema de imágenes ópticas monomodales y redes excesivamente pesadas inviables para UAVs.
* **Soluciones alcanzadas**: Identificación de cuatro tendencias futuras clave: métodos ultraligeros, datasets específicos de misión, fusión de datos multimodales a gran escala, e integración de Vision-Language Models (VLMs) satelitales.
