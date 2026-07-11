# Task-wise Sampling Convolutions for Arbitrary Oriented Object Detection in Aerial Images

- **Key**: Huang2025TSConv
- **Year**: 2025
- **Venue**: arXiv:2209.02200v3 [cs.CV] / IEEE Transactions on Geoscience and Remote Sensing (TGRS)

## Resumen

Este artículo aborda el problema de la Detección de Objetos Arbitrariamente Orientados (Arbitrary-Oriented Object Detection, AOOD) en imágenes de teledetección. El trabajo identifica que los modelos CNN existentes para AOOD sufren de un problema fundamental denominado Inconsistente Sensibilidad de Características (IFS, Inconsistent Feature Sensitivity), que ocurre en dos dimensiones: (1) entre subtareas distintas (localización vs. clasificación tienen regiones sensibles diferentes), y (2) entre distintas orientaciones de los objetos (la región sensible varía con el ángulo). Estos dos IFS se acoplan, complicando aún más el entrenamiento. Para resolver este problema, se propone el método TS-Conv (Task-wise Sampling Convolutions), que incluye: (a) LS-Conv, una convolución para muestrear características de localización desde regiones sensibles a la posición del bounding box orientado (OBB); (b) CS-Conv con un kernel circular dinámico (DCK) para extraer características de clasificación robustas a la orientación; y (c) DTLA (Dynamic Task-consistent-aware Label Assignment), una estrategia de asignación dinámica de etiquetas que selecciona posiciones positivas óptimas según puntuaciones combinadas de localización y clasificación. Experimentos extensivos en DOTA v1.0/v1.5/v2.0, HRSC2016, DIOR-R, DroneVehicle y SSDD+ demuestran que TS-Conv supera a la mayoría de métodos AOOD existentes, logrando mAP50=78.75 en DOTAv1.0 (escala única) y mAP50=80.97 con escala múltiple, con ventaja adicional en velocidad de inferencia y escalabilidad a modelos ligeros y datos multimodales.

## Secciones y Subsecciones

### I. Introduction
La introducción motiva el trabajo presentando el problema del IFS en modelos AOOD. Se explica que mientras en la detección de objetos general (GOD) el IFS entre subtareas es conocido, en AOOD este problema se amplifica pues las OBBs tienen orientación, lo que introduce un segundo IFS entre distintas orientaciones. Además, la distribución densa de objetos en imágenes de teledetección exacerba el IFS al provocar interferencia entre regiones sensibles de objetos adyacentes con convolución estándar. Se presentan tres contribuciones principales: TS-Conv con sus estrategias de muestreo explícitamente supervisadas, la integración de representación OBB y coordenadas espaciales en LS-Conv, el kernel circular dinámico DCK para CS-Conv, y la estrategia de asignación dinámica DTLA.
* **Problemas atacados**: El IFS entre subtareas (localización vs clasificación) y entre orientaciones, y su acoplamiento, que degrada la calidad de las predicciones en AOOD.
* **Limitaciones de ese entonces**: Los métodos previos (RepPoints, Oriented RepPoints) compartían los mismos offsets de muestreo para ambas subtareas, forzando un alineamiento inadecuado. Los métodos que usaban puntuaciones combinadas post-procesamiento tampoco eliminaban el IFS en la extracción de características. Ningún método anterior abordaba el IFS de orientaciones junto al de subtareas.
* **Soluciones alcanzadas**: Propuesta de TS-Conv como framework unificado que aborda simultáneamente el IFS de subtareas, orientaciones, y lo integra con la representación OBB y la asignación dinámica de etiquetas.

### II. Related Works

#### A. Arbitrary-Oriented Object Detection
Revisión de los métodos AOOD existentes categorizados por su estrategia de representación de OBB: predicción de ángulo respecto a HBB (ROI Transform, SCRDet, R3Det, S2ANet), predicción directa de cuatro vértices (Gliding Vertex, GGHL), representación por conjunto de puntos (Oriented RepPoints), clasificación discreta de ángulo (CSL), y métodos anchor-free (BBAVectors, O2-DNet, GGHL, AO2-DETR). Adicionalmente se describen mejoras en funciones de pérdida (GWD, KLD para distribuciones gaussianas) y métodos de asignación (CFA, DCFL).
* **Problemas atacados**: La necesidad de representar y predecir orientaciones arbitrarias de manera precisa y eficiente para objetos en imágenes aéreas.
* **Limitaciones de ese entonces**: Los métodos basados en predicción de ángulo sufren discontinuidades angulares; los basados en vértices tienen ambigüedad; los métodos con clasificación discreta de ángulo tienen resolución limitada.
* **Soluciones alcanzadas**: Diversas representaciones OBB que atacan los problemas de discontinuidad, ambigüedad y precisión desde distintos ángulos.

#### B. Solutions for the IFS Problem
Revisión de los métodos que abordan el IFS en detección de objetos general: IoU-Net (predicción de puntuación de localización adicional), Double-Head R-CNN y YOLOX (cabezas desacopladas), TSD y D2Det (RoI Pooling deformable por tarea), VFNet (alineación de características desacopladas). En AOOD: GGHL y CFC-Net (desacoplamiento y reweighting de características por tarea), Oriented RepPoints (offsets compartidos + aprendizaje de offsets de refinamiento). Se señala que ninguno aborda el IFS de orientaciones junto al de subtareas.
* **Problemas atacados**: Encontrar estrategias de extracción de características que optimicen simultáneamente para localización y clasificación sin sacrificar la calidad en ninguna.
* **Limitaciones de ese entonces**: Los métodos con offsets compartidos fuerzan posiciones de muestreo idénticas para tareas con sensibilidades distintas; los métodos de reweighting no eliminan el IFS en la extracción sino que lo compensan post-hoc.
* **Soluciones alcanzadas**: El paper propone TS-Conv como solución que va más allá del reweighting o refinamiento, diseñando muestreos completamente distintos y explícitamente supervisados para cada subtarea.

### III. Proposed GGHL Framework (TS-Conv)

#### A. Convolution for Sampling Localization Features (LS-Conv)
LS-Conv diseña los 9 puntos de muestreo de una convolución 3×3 para que correspondan a los 9 puntos clave de la representación OBB de GGHL: los 4 vértices del HBB, los 4 vértices del OBB y el centro del candidato. Se usa DCN (deformable convolution) para mover adaptativamente los puntos de muestreo según el OBB predicho inicialmente. Se añade una operación de embeddings de coordenadas espaciales (SCE) que permite a la CNN aprender las coordenadas absolutas de las posiciones muestreadas, haciendo la supervisión de localización más explícita y comprensiva. Los puntos en los cuatro bordes del OBB son deslizantes (controlados por variables aprendibles σᵢ) para adaptarse a la forma real del objeto.
* **Problemas atacados**: La falta de asociación explícita entre las posiciones de muestreo de la CNN y la representación geométrica del OBB, que causaba que la información de localización fuera aprendida de forma implícita.
* **Limitaciones de ese entonces**: Los métodos anteriores como RepPoints y Oriented RepPoints usaban el conjunto de puntos para derivar el bounding box después, lo que no garantizaba que los puntos de muestreo correspondieran a las regiones más informativas para localización.
* **Soluciones alcanzadas**: LS-Conv asocia directamente los puntos de muestreo con los vértices del OBB y las coordenadas espaciales, haciendo que el campo receptivo siempre se adapte al tamaño del objeto y la supervisión sea más precisa.

#### B. Convolution for Sampling Classification Features (CS-Conv)
CS-Conv diseña los puntos de muestreo de clasificación para moverse libremente dentro del MERect (rectángulo externo mínimo del OBB), sin las restricciones rígidas de LS-Conv. Para abordar el IFS de orientaciones, se diseña el Dynamic Circular Kernel (DCK) que genera kernels circulares por interpolación bilineal desde el kernel cuadrado estándar, rotándolos en 8 orientaciones (múltiplos de π/4). Se fusionan adaptativamente los 8 kernels orientados mediante pesos aprendibles βₖ, y se implementa como convolución de grupo para eficiencia. El resultado es un kernel que extrae características robustas a la orientación sin la necesidad de augmentación con rotación aleatoria.
* **Problemas atacados**: La variabilidad de las regiones sensibles de clasificación según la orientación del objeto, que causa que un único kernel de convolución estándar sea subóptimo para objetos de distintas orientaciones.
* **Limitaciones de ese entonces**: Los kernels de convolución estándar no son equivariantes a rotación; las augmentaciones de rotación aleatoria mejoran pero no eliminan la sensibilidad a la orientación.
* **Soluciones alcanzadas**: DCK adapta dinámicamente la orientación y pesos del kernel según el objeto, logrando robustez a orientaciones arbitrarias sin costo extra de augmentación.

#### C. Dynamic Task-consistent-aware Label Assignment (DTLA)
DTLA extiende la estrategia estática GGHL (asignación basada en mapas de calor gaussianos) con una asignación dinámica que selecciona el Top-P posiciones positivas según una puntuación combinada Dₓᵧ que integra el score gaussiano previo Fₓᵧ, la puntuación de localización Lₓᵧ (basada en pérdida GIoU) y la puntuación de clasificación Ĉₓᵧ. El peso entre prior estático y score dinámico cambia durante el entrenamiento (ϑ̃ decrece con las iteraciones). Las posiciones se dividen en: positivas (Top-P según Dₓᵧ), negativas (fuera de la región gaussiana), soft-negativas (en región gaussiana pero no positivas, con peso wₛₙₑg=1−Dₓᵧ), e ignoradas (cuando Fₓᵧ y Dₓᵧ se contradicen).
* **Problemas atacados**: La asignación estática de etiquetas que no puede retroalimentar a la CNN sobre qué posiciones candidatas son óptimas para ambas subtareas simultáneamente, generando ambigüedad en el entrenamiento.
* **Limitaciones de ese entonces**: GGHL usaba umbralización fija sobre el mapa gaussiano; las posiciones más cercanas al centro gaussiano no son necesariamente las mejores para localización y clasificación. Los métodos estáticos no adaptan la selección a las características aprendidas durante el entrenamiento.
* **Soluciones alcanzadas**: DTLA selecciona dinámicamente posiciones positivas óptimas usando scores task-aware alineados espacialmente gracias a TS-Conv, formando un bucle cerrado de "asignación→muestreo→alineamiento→reasignación" durante el entrenamiento.

### IV. Experiments and Discussions

#### A. Experimental Conditions
Describe los datasets utilizados: DOTAv1.0 (188K objetos, 15 categorías, 2806 imágenes), DOTAv2.0 (1.79M objetos, 18 categorías), HRSC2016 (detección de barcos, 1061 imágenes), DIOR-R (190K objetos, 20 categorías, 23463 imágenes), DroneVehicle (datos infrarrojo-RGB para vehículos, ~28K pares), SSDD+ (SAR, 1160 imágenes). Métricas: mAP50, mAP75, mAP50:95. Plataforma: servidor AMD 3950WX + 4 GPUs NVIDIA RTX 3090 + dispositivos embebidos Jetson AGX Xavier y TX2.
* **Problemas atacados**: Evaluar la generalidad y escalabilidad de TS-Conv en escenarios diversos (múltiples escenas, modalidades, categorías, condiciones de iluminación).
* **Limitaciones de ese entonces**: Los métodos previos típicamente se evaluaban solo en DOTA, sin probar en múltiples modalidades (infrarrojo, SAR) ni en dispositivos embebidos.
* **Soluciones alcanzadas**: Protocolo de evaluación comprehensivo que cubre 5+ datasets de distintas características para demostrar la generalidad de TS-Conv.

#### B. Ablation Experiments
Los experimentos ablación sobre DOTAv1.0 validan cada componente de TS-Conv con GGHL como baseline (mAP50=76.95, mAP75=44.19, mAP50:95=44.29). LS-Conv solo mejora mAP75 en +2.19 (el mayor impacto individual), demostrando su importancia para localización precisa. CS-Conv con DCK mejora +1.11 en mAP50. La combinación TS-DCN (LS+CS) logra +1.18/+2.40/+1.80 en mAP50/mAP75/mAP50:95. DTLA añade +0.62 en mAP50 sin costo de inferencia. TS-Conv completo alcanza +1.80/+2.41/+1.98 vs baseline. Las visualizaciones confirman que TS-Conv alinea correctamente las regiones sensibles de localización y clasificación en posiciones espacialmente consistentes.
* **Problemas atacados**: Verificar la contribución independiente de cada componente (LS-Conv, CS-Conv+DCK, DTLA) y sus interacciones sinérgicas.
* **Limitaciones de ese entonces**: Los experimentos comparativos con shared-offset DCNs (como en RepPoints) mostraron que el alineamiento forzado sacrifica la calidad de localización por ligeramente mejorar clasificación.
* **Soluciones alcanzadas**: Cada componente de TS-Conv contribuye de manera complementaria: LS-Conv mejora principalmente localización precisa (mAP75), CS-Conv+DCK mejora robustez a orientaciones, y DTLA integra ambas en una asignación coherente.

#### C. Experiments for the Scalability of TS-Conv

##### 1) Scalability on Lightweight Models
TS-Conv se extiende a modelos ligeros mediante destilación de conocimiento (knowledge distillation). El modelo LO-Det ligero con DTLA mejora +2.10 en mAP50 sin costo de inferencia adicional. TS-Conv Lite (LO-Det distilado desde TS-Conv como teacher) mejora +7.79 sobre el baseline con menos parámetros y mayor velocidad. Alcanza mAP50=73.96 en DOTAv1.0 a 62.07 fps en GPU, superando a YOLOv6nano y YOLOXnano en rendimiento. En dispositivos Jetson (TX2, AGX Xavier, Nano) también muestra ventaja de velocidad y precisión.
* **Problemas atacados**: El alto costo computacional de DCNs que limita la aplicación de TS-Conv en dispositivos embebidos o aplicaciones de tiempo real.
* **Limitaciones de ese entonces**: Los modelos AOOD de alta precisión eran generalmente pesados y lentos, no aptos para dispositivos edge con recursos limitados.
* **Soluciones alcanzadas**: Mediante destilación de conocimiento, los beneficios de TS-Conv (features task-aware) se transfieren a modelos ligeros sin incrementar la complejidad de inferencia.

##### 2) Scalability for Multimodal Data
TS-Conv se extiende para datos multimodales (RGB+infrarrojo) mediante TS-Conv* que muestrea características por modalidad además de por tarea. En DroneVehicle, TS-Conv (solo infrarrojo) logra 71.27% mAP50, y TS-Conv* (RGB+IR) alcanza 72.33%, superando a UA-CMDet (multimodal específico) en 64.01%. Las visualizaciones muestran que RGB e infrarrojo tienen regiones sensibles distintas para localización y clasificación, demostrando la utilidad del muestreo por modalidad.
* **Problemas atacados**: La dificultad de fusionar eficazmente información de múltiples modalidades de imagen con distintas propiedades espectrales y sensibilidades.
* **Limitaciones de ese entonces**: Los métodos multimodales existentes (UA-CMDet) usaban estrategias de fusión más complejas y especializadas que no se basaban en el principio de "muestreo separado, mapeo alineado".
* **Soluciones alcanzadas**: TS-Conv* demuestra que el principio de muestreo separado y mapeo alineado de TS-Conv se generaliza naturalmente a múltiples modalidades, obteniendo mejoras significativas con bajo overhead.

#### D. Comparison Experiments
TS-Conv se compara contra métodos AOOD de punta en múltiples datasets. En DOTAv1.0 (escala única): mAP50=78.75 a 23.23 fps, superando a la mayoría de métodos anchor-free y siendo competitivo con métodos anchor-based más lentos. En escala múltiple: mAP50=80.97 a 16.49 fps. En DOTAv1.5 y v2.0 también supera al baseline GGHL en +2.86 y +2.60 respectivamente. En HRSC2016: mAP50(07)=90.59, mAP75(07)=78.34, mejorando al baseline GGHL en +1.06 y +2.27 respectivamente. En DIOR-R: mAP75=42.69 (+5.70) y mAP50:95=41.38 (+3.94), mostrando ganancias especialmente notables en métricas de alta precisión. En SSDD+ (SAR) también supera a métodos existentes.
* **Problemas atacados**: Demostrar que TS-Conv tiene rendimiento superior y general frente al estado del arte en múltiples escenarios y configuraciones de evaluación.
* **Limitaciones de ese entonces**: Muchos métodos AOOD state-of-the-art son o bien más lentos (métodos de dos etapas o refinamiento) o bien menos precisos (métodos anchor-free de una etapa).
* **Soluciones alcanzadas**: TS-Conv logra un mejor equilibrio velocidad-precisión que los métodos comparados, siendo especialmente notable la mejora en mAP75 y mAP50:95 que indican mayor calidad de localización.

### V. Conclusions and Discussions
Resume las contribuciones: (1) análisis comprehensivo del IFS en AOOD, (2) TS-Conv con muestreos explícitamente supervisados por tarea, (3) DCK para robustez a orientaciones, (4) DTLA dinámica de asignación de etiquetas, (5) escalabilidad a modelos ligeros y datos multimodales. Limitaciones reconocidas: el uso de DCNs incrementa la complejidad computacional frente al baseline, y el impacto del error de anotación OBB en las restricciones explícitas de LS-Conv no está completamente analizado. Se señala como trabajo futuro la investigación de estas limitaciones. El código está disponible en https://github.com/Shank2358.
* **Problemas atacados**: Síntesis de los avances logrados y reconocimiento honesto de las limitaciones que quedan por resolver.
* **Limitaciones de ese entonces**: El costo adicional de DCNs respecto al baseline simple, y la sensibilidad potencial a errores de anotación OBB en las restricciones de LS-Conv.
* **Soluciones alcanzadas**: TS-Conv demuestra efectividad, escalabilidad y generalidad en múltiples datasets y modalidades, con desempeño superior en métricas de calidad de localización (mAP75, mAP50:95) frente al estado del arte.
