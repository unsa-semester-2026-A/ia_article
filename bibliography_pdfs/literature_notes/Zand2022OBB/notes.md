# Oriented Bounding Boxes for Small and Freely Rotated Objects

- **Key**: Zand2022OBB
- **Year**: 2022
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing (TGRS)

## Resumen
Este artículo presenta un método novedoso de detección de objetos diseñado para manejar objetos con rotación libre y tamaños arbitrarios bajo sensores de percepción remota, logrando detectar instancias extremadamente pequeñas de hasta 2 × 2 píxeles. A diferencia de las metodologías tradicionales que tratan la localización de cajas orientadas (OBB) como un problema de regresión continua (añadiendo parámetros como ángulos o coordenadas de esquinas, lo cual introduce discontinuidades de gradiente y ambigüedades), este enfoque reformula la detección como un problema de clasificación discreta pura. Mediante un backbone DarkNet-53 modificado llamado DarkNet-RI (Rotation-Invariant) estructurado en un decodificador piramidal de 5 escalas, la red aprende clasificaciones semánticas a nivel de píxel con restricciones de regularización para garantizar la invarianza a rotaciones de 360°. Las cajas orientadas mínimas se infieren en tiempo de ejecución de manera puramente geométrica a través de operaciones morfológicas y algoritmos de contorno sobre las predicciones de clase conectadas, eliminando la necesidad de definir cajas de anclaje (anchors) y la sobrecarga computacional de regresión. Las pruebas en los datasets xView y DOTA demuestran que el método supera consistentemente a detectores de regresión de vanguardia tanto en precisión como en eficiencia.

## Secciones y Subsecciones

### I. Introduction
Establece la importancia de la detección de objetos en imágenes de percepción remota obtenidas por satélites y drones. Identifica los principales desafíos del dominio aéreo: la presencia de objetos densamente agrupados de tamaños muy dispares (desde vehículos diminutos hasta edificios grandes) y la total ausencia de una orientación preferencial debido a la vista perpendicular del sensor.
* **Problemas atacados**: La incapacidad de los detectores estándar para localizar de forma precisa objetos libres de orientación y de tamaño extremadamente pequeño (p. ej. autos representados por tan solo 2x2 píxeles) sin causar desalineaciones graves en las cajas delimitadoras.
* **Limitaciones de ese entonces**: Los detectores tradicionales basan su diseño en cajas alineadas con los ejes (HBB), que incorporan demasiado ruido de fondo y superposiciones en vecindades densas. Los pocos detectores OBB existentes se apoyan en regresores de ángulos propensos a discontinuidades numéricas en los límites de rango y a configuraciones complejas de anclajes.
* **Soluciones alcanzadas**: Propuesta de un marco CNN basado en clasificación pura libre de cajas de anclaje (anchors) que extrae características invariantes a la rotación en múltiples niveles de escala y determina OBBs únicamente en tiempo de inferencia empleando contornos geométricos.

### II. Related Work
Se analizan los enfoques de detección y segmentación existentes en dos vertientes principales: métodos de detección de objetos de propósito general y detectores específicos para percepción remota.
* **Problemas atacados**: Contextualizar la propuesta de clasificación e invarianza rotacional frente a las soluciones previas de regresión de cajas horizontales y orientadas.
* **Limitaciones de ese entonces**: Los marcos de propósito general (Faster R-CNN, YOLO, SSD) dependen fuertemente de propuestas de región o anclajes horizontales que restringen espacialmente las predicciones en zonas densas. En el ámbito de percepción remota, los modelos OBB (R-DFPN, RRPN, ICN, SCRDet, RoI Transformer) agregan complejidad matemática mediante sub-redes de regresión de ángulo que sufren de la ambigüedad en el orden de las esquinas del cuadrilátero.
* **Soluciones alcanzadas**: Clasificación conceptual de las arquitecturas previas y propuesta de una alternativa modular que elimina por completo la etapa de regresión continua para la orientación espacial.

#### A. General Object Detection Methods
Detalla el funcionamiento de los métodos clásicos de dos etapas (R-CNN, Fast R-CNN, Faster R-CNN) y de una sola etapa (YOLO y variantes).
* **Problemas atacados**: La ineficiencia en el cálculo repetitivo de propuestas y el tratamiento de áreas congestionadas.
* **Limitaciones de ese entonces**: YOLOv3 y detectores similares aplican divisiones de grilla toscas y anclajes con restricciones espaciales estrictas. Dos objetos cuyos centros estén a menos de 32 píxeles de distancia no pueden diferenciarse en la grilla estándar de YOLO, perdiendo objetos pequeños agrupados.
* **Soluciones alcanzadas**: Identificación de la necesidad de una grilla de predicción de mayor resolución (p. ej. decodificador denso) sin depender de propuestas externas o anclajes predefinidos.

#### B. Remote Sensing Object Detection
Revisión de detectores modificados para satélites y drones, incluyendo enfoques con cajas orientadas (OBB).
* **Problemas atacados**: Adaptación a la rotación libre y la resolución espacial masiva de tomas aéreas.
* **Limitaciones de ese entonces**: Los enfoques de OBB existentes (p. ej. RRPN con anclajes rotados o RoI Transformer con transformaciones espaciales) requieren un gran volumen de cómputo adicional debido al cálculo de IoU orientado sobre múltiples anclajes rotados de prueba. Además, la definición del ángulo sufre de ambigüedad de límites.
* **Soluciones alcanzadas**: Desarrollo de una tabla comparativa (Table I) que expone las estrategias de OBB de los competidores e introduce el concepto de modelar la detección mediante clasificación semántica por píxeles y posterior extracción de contornos mínimos.

### III. Proposed Method
Presentación de la arquitectura DarkNet-RI y de las tres fases del pipeline: segmentación semántica multiescala, determinación de cajas orientadas y refinamiento de cajas.
* **Problemas atacados**: Detección conjunta de cajas alineadas y orientadas a múltiples escalas espaciales sin añadir sobrecargas en la función de pérdida por regresión.
* **Limitaciones de ese entonces**: La regresión directa requiere predecir 8 coordenadas o parámetros paramétricos de ángulo propensos a inestabilidad.
* **Soluciones alcanzadas**: Definición de un pipeline integrado que extrae características multiescala utilizando DarkNet-RI, calcula la probabilidad de clase por píxel y extrae las cajas mínimas orientadas usando geometría computacional en inferencia.

#### A. Problem Setup
Formulación matemática de la tarea de predicción de cajas orientadas y la representación del objeto.
* **Problemas atacados**: Representación unificada y robusta de las coordenadas de las cajas de esquina de los objetos.
* **Limitaciones de ese entonces**: La descripción clásica de cajas delimita incorrectamente objetos con rotaciones inclinadas y asume anclajes de proporciones fijas.
* **Soluciones alcanzadas**: Representación de cada caja orientada mediante sus 4 esquinas ordenadas de forma que la orientación quede resuelta implícitamente por la geometría del polígono circunscrito.

#### B. Multi-Scale Semantic Segmentation
Arquitectura de codificador-decodificador basada en DarkNet-53 con skip connections e invarianza de rotación acoplada al entrenamiento.
* **Problemas atacados**: Extracción de características robustas e invariantes a rotaciones en el plano.
* **Limitaciones de ese entonces**: El aumento de datos clásico por rotación es insuficiente para garantizar que la red aprenda características idénticas para diferentes orientaciones. Añadir capas de invarianza de rotación a la red incrementa el riesgo de sobreajuste y la carga computacional.
* **Soluciones alcanzadas**: Creación de DarkNet-RI que conecta el decodificador de 5 niveles con skip connections y una función de pérdida que introduce un término de regularización de invarianza a la rotación, forzando a que las características extraídas de una instancia original y su contraparte rotada sean similares.

##### 1) Pyramid Representation Learning Layer
Estructura piramidal para la estimación a 5 escalas espaciales distintas.
* **Problemas atacados**: Desbalance de escala entre objetos pequeños y grandes en la misma escena aérea.
* **Limitaciones de ese entonces**: Los detectores piramidales estándar evalúan todos los tamaños de objetos en todos los niveles, lo que confunde a las capas finas con objetos gigantes y a las gruesas con objetos minúsculos.
* **Soluciones alcanzadas**: Implementación de una estrategia multiescala "on-off" (Fig. 3) que activa o desactiva la responsabilidad de un objeto en un nivel de escala específico basándose en si su tamaño físico cabe por completo dentro de una sola celda de la grilla de ese nivel.

##### 2) In-Plane Rotation-Invariance
Formulación del término de regularización para forzar representaciones de características consistentes bajo rotaciones de 360 grados.
* **Problemas atacados**: Pérdida de características discriminantes ante inclinaciones arbitrarias de los objetos en tomas aéreas.
* **Limitaciones de ese entonces**: Métodos anteriores agregaban capas matemáticas complejas a la red que degradaban el tiempo de procesamiento.
* **Soluciones alcanzadas**: Diseño de una pérdida de regularización de rotación (Eq. 1) que penaliza la distancia euclidiana entre las representaciones vectoriales del objeto original y su versión rotada aleatoriamente, aplicada únicamente en su área geométrica de traslape (Fig. 5).

##### 3) Training
Definición de la función de pérdida multiobjetivo combinada.
* **Problemas atacados**: Optimización conjunta de la presencia de objetos, clasificación de categorías y la invarianza rotacional.
* **Limitaciones de ese entonces**: Funciones de pérdida inestables cuando se combinan pérdidas de regresión suaves con pérdidas de clasificación.
* **Soluciones alcanzadas**: Función de pérdida compuesta (Eq. 2) que integra una pérdida de confianza por Softmax, una pérdida de clasificación por Entropía Cruzada y la pérdida de regularización rotacional, sumadas en los 5 niveles del decodificador.

#### C. Oriented Box Determination
Algoritmo de inferencia geométrica para generar cajas orientadas mínimas a partir de mapas de clases binarias.
* **Problemas atacados**: Obtención de cajas orientadas precisas sin realizar regresiones paramétricas continuas.
* **Limitaciones de ese entonces**: Los métodos basados en regresión de cajas orientadas sufren de discontinuidades angulares y sobrecostos por evaluar miles de anclajes rotados.
* **Soluciones alcanzadas**: Inferencia puramente geométrica en tiempo de ejecución. La red genera matrices de clase por píxel a las que se aplican operaciones morfológicas de suavizado y denoising. Luego, se extraen los contornos con el algoritmo de Suzuki, se calcula el convex hull y se determina la caja orientada de área mínima mediante el algoritmo de Rotating Calipers, devolviendo $(x, y, w, h, \alpha)$.

#### D. Box Refinement
Estrategia de supresión de no máximos multiescala y filtrado de propuestas de baja calidad.
* **Problemas atacados**: Eliminación de cajas redundantes y falsos positivos en zonas de alta densidad sin eliminar objetos pequeños legítimos vecinos.
* **Limitaciones de ese entonces**: El NMS tradicional con un solo umbral global elimina objetos pequeños legítimos adyacentes a objetos grandes o muy agrupados.
* **Soluciones alcanzadas**: Implementación de un NMS multiescala utilizando umbrales distintos $\{\theta_1, ..., \theta_5\}$ para cada nivel de la pirámide. Además, se refina el criterio de eliminación promediando la confianza de todas las celdas individuales dentro de la caja mínima para descartar propuestas inconsistentes.

### IV. Experiments
Evaluación cuantitativa e implementación sobre los datasets de gran escala xView y DOTA, comparando el rendimiento frente a múltiples detectores de referencia.
* **Problemas atacados**: Validación empírica del rendimiento y la velocidad del método propuesto en escenarios reales de percepción remota.
* **Limitaciones de ese entonces**: Muchos estudios no desglosaban el rendimiento en clases de objetos extremadamente pequeños o densamente poblados.
* **Soluciones alcanzadas**: Implementación del framework en una GPU Titan RTX logrando tiempos de inferencia de 60 ms por imagen en xView y 10 ms en DOTA.

#### A. Datasets and Protocols
Descripción de las especificaciones de xView y DOTA y el protocolo de evaluación comparativa.
* **Problemas atacados**: Normalización del tamaño de las imágenes para el procesamiento en la red.
* **Limitaciones de ese entonces**: Las resoluciones masivas y variables de las imágenes satelitales impiden su introducción directa en redes convolucionales estándar.
* **Soluciones alcanzadas**: Estrategia de segmentación en parches de $512 \times 512$ píxeles con un solapamiento de 10 píxeles y relleno con ceros (zero padding) para mantener las relaciones de aspecto de la imagen original.

#### B. xView Dataset Experiment
Resultados experimentales detallados en xView, enfocándose en la detección de 19 clases de objetos pequeños y similares.
* **Problemas atacados**: Detección de objetos pequeños en condiciones de alto desbalance de clases y alta similitud interclase.
* **Limitaciones de ese entonces**: Los detectores SSD y YOLOv3 fallan ante objetos pequeños en vecindades densas debido a las restricciones de tamaño de celda de la grilla final.
* **Soluciones alcanzadas**: El método propuesto superó a todos los competidores logrando un mAP de 0.3065 en las 19 clases pequeñas y un mAP general de 0.5315 en las 60 categorías completas de xView, demostrando la efectividad de la grilla fina de $256 \times 256$ en el nivel más detallado.

#### C. DOTA Dataset Experiment
Evaluación del desempeño en la tarea de cajas orientadas (OBB) del dataset DOTA.
* **Problemas atacados**: Extracción precisa de cajas orientadas en un conjunto de 15 categorías de objetos aéreos comunes.
* **Limitaciones de ese entonces**: Los detectores con OBB basados en regresión (como SCRDet o RoI Transformer) cometen fallos por ambigüedad angular en objetos con relaciones de aspecto alargadas.
* **Soluciones alcanzadas**: DarkNet-RI alcanzó un mAP del 75.5%, superando a competidores líderes como SARD (72.95%) y RoI Transformer (69.56%), y lideró el rendimiento en 8 categorías individuales.

#### D. Ablation Study
Estudios de ablación para validar de forma aislada las contribuciones del modelamiento multiescala y de invarianza de rotación.
* **Problemas atacados**: Demostrar el impacto individual del término de regularización rotacional y la pirámide de escalas.
* **Limitaciones de ese entonces**: Dificultad para discernir si las mejoras provienen de la arquitectura de la red o de la estrategia de aumento de datos.
* **Soluciones alcanzadas**: Confirmación de que: 1) la regularización de invarianza rotacional supera al aumento de datos simple (0.5315 mAP vs 0.5185 mAP), 2) excluir la pirámide multiescala reduce drásticamente el mAP en clases grandes, y 3) el backbone DarkNet-RI supera a arquitecturas clásicas como UNet y SegNet adaptadas al mismo pipeline.

##### 1) Rotation-Invariant Feature Learning
* **Problemas atacados**: Validar la contribución del término de regularización angular.
* **Limitaciones de ese entonces**: El aumento tradicional de rotaciones no cubre de forma continua y suave la representación de las características.
* **Soluciones alcanzadas**: Demostración de que la inclusión de la pérdida de regularización eleva la precisión de 0.5018 (sin rotación) a 0.5315 mAP en xView.

##### 2) Multi-Scale Feature Representation
* **Problemas atacados**: Determinar la contribución de las capas intermedias del decodificador.
* **Limitaciones de ese entonces**: Falta de justificación para usar múltiples niveles de resolución en clasificación de contornos.
* **Soluciones alcanzadas**: Demostración de que desactivar los niveles de escala intermedios provoca un colapso en el rendimiento (ROC de Fig. 10), afectando severamente a categorías de objetos de escala media y grande.

##### 3) Multi-Scale Semantic Segmentation
* **Problemas atacados**: Evaluar la superioridad de DarkNet-RI sobre arquitecturas de segmentación clásicas.
* **Limitaciones de ese entonces**: La suposición de que cualquier modelo de segmentación semántica produciría el mismo rendimiento al extraer OBBs.
* **Soluciones alcanzadas**: Pruebas comparativas que demuestran que el uso de DarkNet-RI (75.5% mAP) supera significativamente a UNet (55.4%), SegNet (60.4%) y SCAttNet (61.9%) debido a la integración más densa de características y skip connections.

#### E. Limitations
Discusión honesta sobre los casos de fallo detectados en el sistema propuesto.
* **Problemas atacados**: Identificar las debilidades del detector ante instancias muy elongadas o anidadas.
* **Limitaciones de ese entonces**: Ningún detector de OBB está completamente libre de errores de segmentación geométrica en condiciones límite.
* **Soluciones alcanzadas**: Identificación y documentación de tres modos de fallo: 1) fragmentación de cajas en objetos muy largos (puentes) o por división de parches, 2) desalineación leve con el eje mayor del objeto, y 3) omisión de objetos pequeños anidados dentro de clases geográficamente más grandes (p. ej. barcos dentro de puertos).

### V. Conclusion
Resumen de las contribuciones y propuestas para desarrollos futuros.
* **Problemas atacados**: Delinear el futuro de la detección de objetos sin anclajes en percepción remota.
* **Limitaciones de ese entonces**: La excesiva dependencia de los detectores de anclajes e IoU predefinidos que restringen la generalización.
* **Soluciones alcanzadas**: Consolidación del modelo como un detector libre de anclajes y de regresión angular mediante clasificación pura, proponiendo para el futuro el uso de extractores de características jerárquicas para resolver similitudes interclase muy finas.
