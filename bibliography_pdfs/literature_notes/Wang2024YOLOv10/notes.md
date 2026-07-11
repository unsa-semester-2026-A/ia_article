# YOLOv10: Real-Time End-to-End Object Detection

- **Key**: Wang2024YOLOv10
- **Year**: 2024
- **Venue**: Neural Information Processing Systems (NeurIPS)

## Resumen
YOLOv10 presenta una nueva generación de detectores de objetos en tiempo real y extremo a extremo (NMS-free). Tradicionalmente, la serie YOLO depende del postprocesamiento mediante supresión de no máximos (NMS) debido al uso de asignaciones de etiquetas de uno a muchos (one-to-many) durante el entrenamiento, lo que añade latencia y dificulta el despliegue óptimo. YOLOv10 resuelve esto proponiendo una estrategia de Asignación Dual Consistente (*Consistent Dual Assignments*) para entrenamiento libre de NMS, combinando una rama "uno a muchos" para una supervisión rica durante el entrenamiento con una rama "uno a uno" libre de NMS para una inferencia eficiente. Además, propone una estrategia integral de diseño de modelo guiada por eficiencia y precisión, optimizando varios componentes: un cabezal de clasificación ligero, submuestreo desacoplado de canal espacial, diseño de bloques basado en rango numérico utilizando el bloque invertido compacto (CIB), convoluciones de núcleo grande y el módulo de atención parcial auto-atendida (PSA). YOLOv10 logra rendimiento y eficiencia del estado del arte en el dataset COCO.

## Secciones y Subsecciones

### 1. Introducción
Presenta el contexto de la detección de objetos en tiempo real y el dominio de la familia YOLO. Explica el cuello de botella que representa el postprocesamiento NMS y la redundancia computacional en la arquitectura física de los modelos anteriores de YOLO.
* **Problemas atacados**: La latencia introducida por el algoritmo NMS durante la inferencia y las ineficiencias/redundancias estructurales presentes en el diseño clásico de la arquitectura YOLO.
* **Limitaciones de ese entonces**: El uso de asignación de etiquetas de uno a muchos (TAL) requiere NMS para limpiar predicciones redundantes, lo que hace que la velocidad sea sensible a hiperparámetros y lenta en hardware de despliegue. Alternativas como DETR tienen un costo computacional de codificador/decodificador que supera a los detectores convolucionales, y los intentos previos de CNN sin NMS degradan la precisión.
* **Soluciones alcanzadas**: Introducción de YOLOv10, un detector en tiempo real y end-to-end que elimina la necesidad de NMS usando Asignación Dual Consistente e introduce un rediseño de arquitectura guiado por eficiencia y precisión.

### 2. Trabajo Relacionado
Revisa la literatura sobre detectores de objetos en tiempo real (familia YOLO) y detectores end-to-end (DETRs y alternativas CNN libres de NMS).
* **Problemas atacados**: Organización histórica de las arquitecturas de la familia YOLO y métodos para evitar componentes manuales de postprocesamiento.
* **Limitaciones de ese entonces**: Los detectores tradicionales basan sus cabezas en anclajes (anchors) y sufren de acumulación de capas costosas en el cuello (neck). Los DETRs eliminan NMS con pérdida de correspondencia húngara, pero su costo computacional de atención cuadrática es prohibitivo para tiempo real extremo.
* **Soluciones alcanzadas**: Contextualización de la combinación híbrida de convoluciones CNN eficientes con autopercepción global optimizada y correspondencia uno a uno diferenciable.

### 3. Metodología
Presenta las dos grandes innovaciones técnicas de YOLOv10: la Asignación Dual Consistente y el Diseño de Modelo guiado por Eficiencia y Precisión.

#### 3.1. Asignaciones Duales Consistentes para Entrenamiento Libre de NMS
Explica cómo se implementan dos cabezales durante el entrenamiento y la métrica de concordancia unificada.
* **Problemas atacados**: La debilidad de supervisión que sufren los modelos cuando se entrenan exclusivamente con correspondencia uno a uno (one-to-one), lo que perjudica la precisión y velocidad de convergencia.
* **Limitaciones de ese entonces**: Los modelos CNN que eliminan NMS usando solo pérdidas uno a uno no logran competir en precisión con sus contrapartes tradicionales que usan asignación uno a muchos (como en YOLOv8).
* **Soluciones alcanzadas**: Incorporación de dos cabezales que operan en paralelo durante el entrenamiento: uno con asignación uno a muchos para brindar rica supervisión al backbone/neck, y otro con asignación uno a uno que no requiere NMS. Durante la inferencia, se descarta el primer cabezal. Para evitar conflictos entre ambos objetivos de clasificación, se propone una métrica de coincidencia consistente ($m_{o2o} = m_{o2m}^r$) que alinea matemáticamente los mejores ejemplos positivos para ambos cabezales.

#### 3.2. Diseño de Modelo Holístico Guiado por Eficiencia y Precisión
Detalla el rediseño del esqueleto neuronal de YOLO para balancear velocidad y capacidad de representación.

##### 3.2.A. Diseño Guiado por Eficiencia
Optimiza el cabezal de clasificación, las capas de downsampling y los bloques internos.
* **Problemas atacados**: El desperdicio de parámetros y operaciones de coma flotante (FLOPs) en componentes redundantes de la red.
* **Limitaciones de ese entonces**: Las cabezas de clasificación y regresión comparten la misma arquitectura costosa a pesar de que la clasificación tiene menos impacto en el cuello de botella del rendimiento. El downsampling tradicional con convolución estándar de $3\times3$ y stride 2 mezcla la reducción espacial con el incremento de canales, resultando caro. Además, aplicar bloques homogéneos en todas las etapas es subóptimo.
* **Soluciones alcanzadas**:
  1. *Cabezal de Clasificación Ligero*: Empleo de convoluciones separables en profundidad de $3\times3$ seguidas de una convolución $1\times1$.
  2. *Downsampling Desacoplado*: Separación en una convolución de punto (pointwise) para modular canales y convolución en profundidad (depthwise) con stride 2 para reducir dimensión espacial.
  3. *Diseño de Bloques Guiado por Rango*: Análisis de redundancia por rango numérico singular de cada etapa. Reemplazo adaptativo de bloques redundantes por el Bloque Invertido Compacto (CIB) que reduce costos de mezcla espacial y de canal.

##### 3.2.B. Diseño Guiado por Precisión
Incrementa el campo receptivo y la atención global en etapas clave.
* **Problemas atacados**: El límite de capacidad de representación local de las convoluciones tradicionales frente a la percepción global de los transformers.
* **Limitaciones de ese entonces**: Agregar mecanismos de auto-atención en todas las etapas resulta inviable en tiempo real por el coste computacional cuadrático de la atención espacial. Las convoluciones de núcleo grande aplicadas a capas tempranas contaminan características de objetos pequeños y aumentan el retardo I/O.
* **Soluciones alcanzadas**:
  1. *Convolución de Núcleo Grande*: Uso de kernels de $7\times7$ en bloques CIB solo en etapas profundas de modelos pequeños, con reparametrización estructural en entrenamiento.
  2. *Atención Parcial Auto-Atendida (PSA)*: División del mapa de canales a la mitad; procesamiento de una mitad mediante cabezales de auto-atención multi-cabeza (MHSA) con BatchNorm y alimentación directa de la otra mitad para posterior fusión.

### 4. Experimentos
Presenta los resultados empíricos y análisis de ablación de los componentes en la tarea de detección del dataset COCO.

#### 4.1. Detalles de Implementación
Describe las directrices del entrenamiento desde cero.
* **Problemas atacados**: Proveer una línea base uniforme para comparar justamente con YOLOv8.
* **Limitaciones de ese entonces**: Sesgos de rendimiento debidos a diferentes optimizadores, aumentos de datos o infraestructuras de inferencia.
* **Soluciones alcanzadas**: Entrenamiento por 500 épocas usando SGD en 8 GPUs NVIDIA 3090, con aumentos Mosaic, Mixup y Copy-Paste. Medición de latencia de inferencia en GPU T4 con TensorRT FP16 y batch size de 1.

#### 4.2. Comparación con el Estado del Arte
Evalúa la eficiencia de YOLOv10 en múltiples escalas (N, S, M, B, L, X) frente a YOLOv6, Gold-YOLO, YOLOv8, YOLOv9 y RT-DETR.
* **Problemas atacados**: Demostrar la superioridad de YOLOv10 en la frontera precisión-velocidad.
* **Limitaciones de ese entonces**: Modelos previos libres de NMS (como RT-DETR) ocupan hasta el triple de parámetros para precisiones similares.
* **Soluciones alcanzadas**: YOLOv10-S es 1.8 veces más rápido que RT-DETR-R18 con la misma precisión y 2.8 veces menos parámetros. YOLOv10-B reduce un 46% la latencia comparado con YOLOv9-C. Todos los modelos superan las versiones equivalentes de YOLOv8 en AP con reducciones masivas de parámetros (p. ej., YOLOv10-M usa un 41% menos parámetros que YOLOv8-M).

#### 4.3. Análisis del Modelo (Ablaciones)
Estudia la contribución individual de cada cambio arquitectónico y estratégico.
* **Problemas atacados**: Validar la necesidad científica de cada componente propuesto.
* **Limitaciones de ese entonces**: Complejidad para interpretar qué modificaciones causan las mejoras observadas en las métricas agregadas.
* **Soluciones alcanzadas**: 
  - La asignación dual consistente reduce la latencia de inferencia en 4.63 ms en YOLOv10-S eliminando NMS.
  - El desacoplamiento espacial-canal de downsampling mejora un 0.7% el AP sobre el downsampling directo disminuyendo la pérdida de información.
  - El CIB con rango adaptativo reduce parámetros de forma masiva sin degradar el AP.
  - El análisis de similitud coseno muestra que los modelos grandes desarrollan características más discriminativas que benefician intrínsecamente la correspondencia libre de NMS.

### 5. Conclusión
Resume las contribuciones del trabajo.
* **Problemas atacados**: Limitaciones de velocidad de inferencia de extremo a extremo e ineficiencias computacionales en detectores en tiempo real.
* **Limitaciones de ese entonces**: Dependencia del postprocesamiento manual de NMS y diseños homogéneos redundantes.
* **Soluciones alcanzadas**: Desarrollo de YOLOv10 como un detector de objetos en tiempo real altamente optimizado que elimina por completo el NMS y redefine los bloques estructurales mediante criterios de eficiencia-precisión.
