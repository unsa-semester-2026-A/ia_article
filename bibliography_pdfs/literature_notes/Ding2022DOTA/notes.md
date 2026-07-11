# Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges

- **Key**: Ding2022DOTA
- **Year**: 2022
- **Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

## Resumen

Este artículo presenta DOTA (Dataset of Object deTection in Aerial images), el dataset más grande de detección de objetos en imágenes aéreas disponible públicamente al momento de su publicación. La motivación central del trabajo es que el progreso en detección de objetos para imágenes naturales no se trasladaba eficientemente a imágenes aéreas, principalmente por la falta de un benchmark de gran escala con anotaciones de *oriented bounding boxes* (OBB). DOTA-v2.0, la versión expandida presentada, contiene 1,793,658 instancias en 18 categorías, anotadas con OBBs y recolectadas de 11,268 imágenes aéreas provenientes de múltiples fuentes (Google Earth, satélites GF-2 y JL-1, e imágenes aéreas CycloMedia). Además del dataset, los autores construyen una librería de código unificada basada en MMDetection para detección orientada, evalúan más de 70 configuraciones de 10 algoritmos de detección y proveen un servidor de evaluación en línea. Los resultados demuestran que el diseño de detectores para imágenes aéreas difiere significativamente del de imágenes naturales (e.g., número óptimo de propuestas mucho mayor, beneficio extremo de aumentos de datos rotacionales y multi-escala). El dataset y los desafíos organizado han atraído más de 1300 equipos de investigación a nivel mundial.

## Secciones y Subsecciones

### 1. Introduction

La introducción contextualiza el problema de detección de objetos en imágenes aéreas (ODAI). Las imágenes aéreas presentan dificultades únicas: orientaciones arbitrarias de los objetos, variaciones extremas de escala, densidades no uniformes y relaciones de aspecto muy grandes. El principal diferenciador respecto a las imágenes naturales es la vista cenital, que hace que los objetos puedan aparecer en cualquier ángulo, invalidando el supuesto de orientación preferencial que tienen los detectores entrenados con imágenes de escenas naturales. Se señala explícitamente la carencia de datasets a gran escala con anotaciones OBB como el principal obstáculo para el avance del área.

* **Problemas atacados**: Ausencia de un benchmark de gran escala con anotaciones OBB para imágenes aéreas; inadecuación de los detectores de imágenes naturales para escenas aéreas.
* **Limitaciones de ese entonces**: Los datasets existentes (e.g., xView, NWPU, VEDAI, DLR 3K) tenían un número limitado de instancias, usaban anotaciones HBB en lugar de OBB, o no cubrían la diversidad de sensores y condiciones del mundo real.
* **Soluciones alcanzadas**: Presentación de DOTA como el dataset de mayor escala para Earth vision, con OBBs, 18 categorías, multi-fuente y ~1.8 millones de instancias.

### 2. Related Work

#### 2.1 Datasets for Conventional Object Detection

Se revisan los principales benchmarks de detección de objetos en imágenes naturales: PASCAL VOC (2005–2012, 20 clases, ~27K bounding boxes), ImageNet (200 clases, ~478K instancias) y MS COCO (91 categorías, 2.5M objetos). Se establece una comparación cuantitativa con DOTA en términos de número de imágenes, área de píxeles, número de instancias y promedio de instancias por imagen, mostrando que DOTA-v2.0 supera a todos en cantidad promedio de instancias por imagen (159.18 vs. 7.19 de COCO).

* **Problemas atacados**: Necesidad de contextualizar DOTA frente a los benchmarks más conocidos del área para justificar su relevancia y escala.
* **Limitaciones de ese entonces**: Los datasets de imágenes naturales no modelan la distribución real de objetos en imágenes aéreas (orientaciones, densidades, tamaños diminutos).
* **Soluciones alcanzadas**: DOTA alcanza una escala comparable a COCO e ImageNet en número de instancias, con la particularidad de tener un promedio de instancias por imagen mucho más alto.

#### 2.2 Datasets for Object Detection in Aerial Images

Se revisan datasets aeroespaciales previos: TAS, SZTAKI-INRIA, NWPU VHR-10, VEDAI, DLR 3K, UCAS-AOD, HRSC2016, xView, VisDrone, DIOR, iSAID, entre otros. Se argumenta que la mayoría son de categoría única o limitada, usan HBB en lugar de OBB, o provienen de una única fuente, lo que introduce sesgos de dominio. Se define formalmente el conjunto de propiedades deseables de un buen dataset para ODAI: (1) datos anotados suficientes, (2) imágenes grandes con información contextual, (3) anotaciones OBB precisas y (4) balance en las fuentes de imágenes.

* **Problemas atacados**: Brecha entre los datasets aéreos disponibles y los requisitos reales para entrenar detectores profundos robustos.
* **Limitaciones de ese entonces**: Datasets como xView y DIOR usan solo HBBs; datasets como NWPU solo tienen 800 imágenes; ninguno combinaba multi-fuente, OBB y escala masiva.
* **Soluciones alcanzadas**: DOTA-v2.0 cumple todas las propiedades deseadas: multi-fuente (Google Earth, GF-2, JL-1, CycloMedia), OBB, 18 categorías y ~1.8M instancias.

#### 2.3 Deep Models for Object Detection in Aerial Images

Se revisan los principales métodos de detección profunda adaptados a imágenes aéreas. Se mencionan: (a) métodos de representación invariante a la rotación (ORN, RRD), (b) métodos que explotan anotaciones OBB explícitamente (R-RCNN con RRoI pooling, RoI Transformer, S2A-Net), (c) métodos que resuelven ambigüedades de definición de OBB (Gliding Vertex, CSL, Mask OBB, CenterMap), y (d) métodos para manejar imágenes aéreas de gran tamaño mediante particionado en parches. Se destaca que la ambigüedad en la definición del OBB (cuatro permutaciones posibles de los vértices de un cuadrilátero) es un reto algorítmico abierto.

* **Problemas atacados**: Adaptación de detectores de imágenes naturales a la detección orientada en imágenes aéreas, especialmente ante variaciones de orientación, escala y densidad.
* **Limitaciones de ese entonces**: Los detectores genéricos no incorporan invarianza a la rotación; las librerías de código existentes (MMDetection, Detectron) no soportaban detección orientada. La ambigüedad de la representación OBB generaba inestabilidades en el entrenamiento.
* **Soluciones alcanzadas**: Revisión sistemática del estado del arte; identificación de los módulos clave a evaluar en los baselines.

#### 2.4 Code Libraries for Object Detection

Se repasan las librerías populares de detección de objetos: TensorFlow Detection API, Detectron, MaskRCNN-Benchmark, Detectron2, MMDetection y SimpleDet. Se señala que estas librerías diseñadas con arquitectura modular facilitan el desarrollo, pero casi ninguna soportaba detección orientada al momento del trabajo (solo Detectron2 tenía soporte limitado). La disparidad en hardware y configuración entre distintas implementaciones dificulta comparaciones justas.

* **Problemas atacados**: Falta de una librería unificada para comparar algoritmos de detección orientada en condiciones controladas.
* **Limitaciones de ese entonces**: MMDetection y similares carecían de operadores críticos para OBB (RoI Align rotado, NMS rotado, cabezas de regresión de ángulo).
* **Soluciones alcanzadas**: Se extiende MMDetection con los módulos necesarios para detección orientada y se usa como librería base para todos los baselines del paper.

### 3. Construction of DOTA

#### 3.1 Image Collection

Las imágenes de DOTA-v2.0 se recolectan de cuatro fuentes: Google Earth, satélites GF-2 y JL-1, e imágenes aéreas de CycloMedia (Rotterdam). Para Google Earth se seleccionaron regiones de interés en todo el mundo (aeropuertos, puertos, áreas urbanas) y se obtuvieron imágenes de 800×800 a 4000×4000 píxeles. Las imágenes de GF-2 y JL-1 se conservan en su tamaño original (hasta ~29,200×27,620 píxeles). Se incluyen tanto vistas nadir como vistas oblicuas (~45°). Esta diversidad de fuentes minimiza el sesgo de dominio del dataset.

* **Problemas atacados**: Sesgo de dominio por fuente única en datasets previos; falta de imágenes de baja densidad de objetos que representen la distribución real del mundo.
* **Limitaciones de ese entonces**: Datasets anteriores usaban principalmente Google Earth, creando un sesgo hacia escenas con alta densidad de objetos de interés.
* **Soluciones alcanzadas**: Combinación de cuatro fuentes de diferente resolución, perspectiva y distribución de objetos, logrando una distribución más fiel a aplicaciones reales.

#### 3.2 Category Selection

Se seleccionan 18 categorías: avión, barco, tanque de almacenamiento, diamante de béisbol, cancha de tenis, piscina, pista de atletismo, puerto, puente, vehículo grande, vehículo pequeño, helicóptero, rotonda, campo de fútbol, cancha de baloncesto, grúa de contenedor, aeropuerto y helipuerto. Las primeras 10 categorías son comunes en datasets previos; las adicionales se eligen por su relevancia en aplicaciones del mundo real (e.g., "helicóptero" como objeto en movimiento, "rotonda" para análisis vial). Se incluyen categorías "stuff" (puerto, aeropuerto) cuyas fronteras son relativamente definibles y que aportan información contextual.

* **Problemas atacados**: Necesidad de cubrir un espectro amplio y representativo de objetos relevantes para aplicaciones reales de Earth vision.
* **Limitaciones de ese entonces**: Datasets previos cubrían pocas categorías o solo objetos de una clase específica (vehículos, barcos, edificios).
* **Soluciones alcanzadas**: 18 categorías que van desde objetos pequeños (vehículo pequeño) hasta objetos tipo "stuff" (aeropuerto), con criterios de selección basados en frecuencia de aparición y valor aplicado.

#### 3.3 Oriented Object Annotation

El proceso de anotación usa OBBs representadas como cuatro vértices {(xi, yi) | i=1,2,3,4} en orden horario. Para facilitar la anotación precisa, los anotadores hacen clic en los cuatro esquinas físicas del objeto o en 4 puntos clave cuando la forma del objeto difiere de un rectángulo (e.g., para aviones se usan: cabeza, dos puntas de ala y cola, luego se convierte al OBB mínimo). Se define un protocolo para resolver la ambigüedad de los cuatro posibles ordenes de vértices: para objetos con cabeza/cola distinguible (vehículos, helicópteros), el primer punto indica la "cabeza"; para objetos sin referencia visual de orientación, se elige el punto superior-izquierdo. El pipeline incluye anotadores expertos, voluntarios en grupos "junior" y "senior", y revisión doble.

* **Problemas atacados**: Ambigüedad en la representación de OBBs (cuatro permutaciones de vértices para el mismo objeto); dificultad de anotación precisa sin referencias claras.
* **Limitaciones de ese entonces**: Las herramientas de anotación previas orientadas a HBBs no eran eficientes para OBBs; la ambigüedad no tenía un protocolo estándar de resolución.
* **Soluciones alcanzadas**: Herramienta de anotación personalizada con clic en esquinas físicas; protocolo claro para el orden de puntos; proceso de revisión doble con expertos en teledetección.

### 4. Properties of DOTA

#### 4.1 Image Sources

DOTA-v2.0 contiene imágenes de tres fuentes: Google Earth (10,186 imágenes, alta densidad de objetos), GF&JL satélite (516 imágenes, baja densidad, más representativas del mundo real) y CycloMedia aérea (566 imágenes, muy baja densidad, ratio de fondo elevado). El ratio de foreground (objetos/imagen) va de 0.003 (satelitales) a 0.037 (Google Earth). Las imágenes de GF-2 y JL-1 son en escala de grises (banda pancromática de 10 bits convertida a 8 bits), mientras que Google Earth y CycloMedia son RGB. Los metadatos de fecha de adquisición están disponibles para todas las imágenes de GF-2, JL-1 y CycloMedia, y para el 27% de las de Google Earth.

* **Problemas atacados**: Sesgo por fuente única y por sobre-representación de escenas densas en objetos.
* **Limitaciones de ese entonces**: Datasets anteriores dominados por imágenes de Google Earth, que son seleccionadas precisamente porque contienen muchos objetos, no representando la distribución real.
* **Soluciones alcanzadas**: Incorporación de imágenes satelitales y aéreas con muy baja densidad de objetos, mejorando la representatividad del dataset para aplicaciones reales.

#### 4.2 Spatial Resolution Information

La resolución espacial (*Ground Sample Distance*, GSD) varía ampliamente en DOTA-v2.0: de 0.1 a 4.5 m/pixel en Google Earth, 0.81 m/pixel para GF-2, 0.72 m/pixel para JL-1, y 0.1 m/pixel para CycloMedia. Solo el 30% de las imágenes tienen GSD disponible, pero esto no es crítico ya que se puede estimar con métodos de aprendizaje. El GSD puede ser útil para filtrar outliers, mejorar clasificación y realizar normalización de escala en los detectores.

* **Problemas atacados**: La falta de información de resolución espacial impide explotar priors de tamaño físico de objetos en los detectores.
* **Limitaciones de ese entonces**: La mayoría de datasets aéreos no incluían información de GSD; los detectores no la aprovechaban.
* **Soluciones alcanzadas**: Provisión de GSD para el 30% de las imágenes; análisis estadístico de la distribución de GSD; discusión de aplicaciones potenciales en detectores.

#### 4.3 Various Instance Orientations

Los objetos en imágenes aéreas tienen distribución uniforme de orientaciones en [-π, π], a diferencia de objetos en escenas naturales (texto, caras) que tienen sesgo gravitacional hacia [-π/2, π/2]. Esta propiedad única de DOTA lo hace especialmente valioso para investigar extracción de características invariante a la rotación y detección de objetos orientados.

* **Problemas atacados**: Los detectores basados en características no invariantes a la rotación fallan ante la distribución uniforme de ángulos en imágenes aéreas.
* **Limitaciones de ese entonces**: Benchmarks de texto y caras no capturaban la variedad completa de orientaciones presente en imágenes aéreas.
* **Soluciones alcanzadas**: DOTA ofrece un benchmark con distribución de orientaciones verdaderamente uniforme, ideal para evaluar métodos de detección orientada.

#### 4.4 Various Instances Pixel Sizes

El 77% de las instancias en DOTA-v2.0 son de tamaño pequeño (10–50 píxeles), el 22% medianas (50–300) y solo el 1% grandes (>300). Esto contrasta con PASCAL VOC donde el 61% son medianas. Las variaciones de escala son extremas tanto dentro como entre categorías, haciendo el dataset particularmente desafiante para detectores basados en anclas con configuración fija.

* **Problemas atacados**: Detección de objetos de tamaño muy pequeño y manejo de variaciones extremas de escala en imágenes aéreas.
* **Limitaciones de ese entonces**: Los detectores optimizados para imágenes naturales no manejaban bien la alta proporción de objetos diminutos (< 10 píxeles) en imágenes aéreas.
* **Soluciones alcanzadas**: DOTA-v1.5 y v2.0 incluyen anotaciones de objetos diminutos (< 10 píxeles) ausentes en v1.0, proporcionando un benchmark más completo para investigación en detección de pequeños objetos.

#### 4.5 Various Instance Aspect Ratios (ARs)

DOTA presenta distribuciones de ARs tanto para OBBs como para HBBs. Muchas instancias tienen ARs elevadas (e.g., puentes, barcos), lo que guía el diseño de anclas en detectores como Faster R-CNN y YOLO. Las relaciones de aspecto de los HBBs generados a partir de OBBs son generalmente mayores que las de los OBBs originales, reflejando la "distorsión" que introduce la representación horizontal en objetos orientados.

* **Problemas atacados**: El diseño adecuado de anclas para detectores depende de la distribución de ARs; anclas mal diseñadas degradan el rendimiento.
* **Limitaciones de ese entonces**: Los detectores usaban distribuciones de ARs derivadas de imágenes naturales que no se adaptan a la distribución de objetos aéreos.
* **Soluciones alcanzadas**: Análisis cuantitativo de la distribución de ARs (OBB y HBB) en DOTA que guía el diseño de anclas para detectores orientados.

#### 4.6 Various Instance Densities of the Images

El número de instancias por parche de 1024×1024 varía enormemente en DOTA: desde 1 hasta más de 1000. Las categorías más densas son "storage tank", "ship" y "small vehicle" (distancias al vecino más cercano < 10 píxeles). Esta variación de densidad es mucho mayor que en cualquier dataset de imágenes naturales, exigiendo más propuestas por imagen y configuraciones de NMS adaptadas.

* **Problemas atacados**: Los hiperparámetros estándar (número de propuestas, umbrales de NMS) optimizados para imágenes naturales son inadecuados para imágenes aéreas.
* **Limitaciones de ese entonces**: Detectores configurados para 300 propuestas (óptimo en PASCAL VOC) no podían capturar los miles de instancias posibles en imágenes aéreas.
* **Soluciones alcanzadas**: Análisis de densidad por categoría; experimentos que muestran que 8000 propuestas es el óptimo para DOTA (vs. 300 en PASCAL VOC).

#### 4.7 DOTA Versions

##### 4.7.1 DOTA-v1.0
Primera versión con 15 categorías, 2,806 imágenes y 188,282 instancias. Los objetos diminutos (< 10 píxeles) no están anotados y las imágenes provienen principalmente de Google Earth. División: 50% entrenamiento, 1/6 validación, 1/3 prueba.

* **Problemas atacados**: Necesidad de un primer benchmark a escala suficiente para entrenar redes profundas para ODAI.
* **Limitaciones de ese entonces**: No incluía objetos diminutos; sesgo hacia fuente única (Google Earth); no cubría escenas con baja densidad de objetos.
* **Soluciones alcanzadas**: Primer dataset público de escala suficiente para entrenar detectores profundos para imágenes aéreas sin necesidad de preentrenamiento en COCO/ImageNet.

##### 4.7.2 DOTA-v1.5
Usa las mismas imágenes que v1.0 pero añade anotaciones de objetos extremadamente pequeños (< 10 píxeles) y agrega la categoría "container crane". Total: 402,089 instancias.

* **Problemas atacados**: Cobertura de objetos diminutos que son críticos en aplicaciones reales (vehículos pequeños, helipads).
* **Limitaciones de ese entonces**: v1.0 ignoraba objetos sub-10 píxeles, subestimando la dificultad real del problema.
* **Soluciones alcanzadas**: Extensión de anotaciones y adición de una categoría industrial relevante, sirviendo como base del desafío DOAI 2019 de CVPR.

##### 4.7.3 DOTA-v2.0
Versión más completa: 18 categorías, 11,268 imágenes (4× más que v1.0), 1,793,658 instancias (~9.5× más que v1.0). Añade categorías "airport" y "helipad". Incorpora imágenes de GF-2 y CycloMedia para mayor diversidad. Dividido en entrenamiento, validación, test-dev y test-challenge (similar a COCO).

* **Problemas atacados**: Necesidad de un dataset que cubra objetos en imágenes de gran tamaño (>20,000×20,000 píxeles) con baja densidad, y que tenga split de test-challenge para competiciones controladas.
* **Limitaciones de ese entonces**: v1.5 aún usaba solo imágenes de Google Earth, sin incluir las imágenes GF-2 y aéreas de alta resolución con distribución real de objetos.
* **Soluciones alcanzadas**: Dataset multi-fuente, multi-resolución, con el mayor número de instancias anotadas con OBB en la historia del Earth vision al momento de publicación.

### 5. Benchmarks

#### 5.1 Evaluation Tasks and Metrics

Se definen dos tareas: detección con HBB y detección con OBB. La métrica principal es mAP (mean Average Precision) usando el protocolo PASCAL VOC 07 (área bajo la curva precisión-recall de 0 a 1). El IoU para OBBs se calcula entre polígonos convexos; el área de intersección se computa descomponiendo los polígonos en triángulos. Se provee código para computar IoU de OBBs tanto en CPU como GPU.

* **Problemas atacados**: Definición de métricas estandarizadas para evaluar detección orientada de manera comparable y reproducible.
* **Limitaciones de ese entonces**: No existía un protocolo de evaluación estándar para OBBs; el IoU entre polígonos orientados es computacionalmente no trivial.
* **Soluciones alcanzadas**: Protocolo basado en PASCAL VOC 07 mAP con cómputo de IoU entre OBBs; código abierto CPU/GPU.

#### 5.2 Implementation Details

Todos los algoritmos se implementan en una librería unificada derivada de MMDetection. Las imágenes grandes se recortan en parches de 1,024×1,024 con stride de 824. Durante inferencia, los resultados de los parches se mapean de vuelta a coordenadas de imagen original y se aplica NMS global. Se usan 4 GPUs con batch total de 8; lr=0.01; esquema "1×" de Detectron (excepto RetinaNet que usa "2×").

* **Problemas atacados**: Comparación justa entre métodos usando el mismo hardware, software y configuración.
* **Limitaciones de ese entonces**: Comparaciones previas en DOTA-v1.0 usaban implementaciones heterogéneas con distintas configuraciones, dificultando la interpretación de diferencias de rendimiento.
* **Soluciones alcanzadas**: Evaluación de 10 algoritmos y más de 70 configuraciones bajo condiciones controladas idénticas.

##### 5.2.1 Baselines with HBBs
Se usan dos estrategias: (1) predicción directa de HBBs con RetinaNet, Mask R-CNN, Cascade Mask R-CNN, Hybrid Task Cascade y Faster R-CNN; (2) conversión de OBBs predichos a HBBs como resultado alternativo.

##### 5.2.2 Baselines with OBBs
Dos enfoques: **OBB Head** (reemplaza la cabeza de regresión de HBB para regresionar OBBs como (x,y,w,h,θ), con selección del mejor GT entre cuatro formas posibles) y **Mask Head** (trata la OBB como una máscara gruesa de nivel de píxel, usando Mask R-CNN). Se evalúan Faster R-CNN OBB, RetinaNet OBB, Faster R-CNN OBB + Dpool, Faster R-CNN OBB + RoI Transformer y variantes.

* **Problemas atacados**: Ausencia de baselines comparables bajo condiciones controladas para detección orientada.
* **Limitaciones de ese entonces**: Las librerías existentes no permitían predicción de OBBs; implementar correctamente la selección del mejor GT y la regresión de ángulo requería modificaciones no triviales.
* **Soluciones alcanzadas**: Dos enfoques complementarios (regresión de OBB vs. clasificación por máscara) con análisis detallado de ventajas y desventajas en cada dataset.

#### 5.3 Codebase and Development Kit

La librería de código extiende MMDetection con: OBB Head, rotated RoI Align, rotated position-sensitive RoI Align y soporte para RRPN y RoI Transformer. El development kit incluye: carga/visualización de GTs, cómputo de IoU entre OBBs (Python/C, CPU y GPU), evaluación de resultados, y herramientas de particionado y fusión de imágenes grandes.

* **Problemas atacados**: Reproducibilidad e implementación eficiente de algoritmos de detección orientada.
* **Limitaciones de ese entonces**: No existía una librería pública que integrara todos los operadores necesarios para ODAI orientada.
* **Soluciones alcanzadas**: Librería modular compatible con MMDetection, pública en GitHub, que facilita la implementación de nuevos algoritmos de detección orientada.

### 6. Results

#### 6.1 Benchmark Results and Analyses

Los resultados muestran una caída progresiva de mAP de DOTA-v1.0 (hasta 73.76%) a v1.5 (65.03%) a v2.0 (52.81%), confirmando el aumento de dificultad. En todos los datasets, Faster R-CNN OBB + RoI Transformer es el mejor método en la curva velocidad-precisión. El Mask Head converge más fácilmente pero es más lento; OBB Head es más rápido. Los resultados de OBB mAP son ligeramente menores que los de HBB mAP para el mismo detector, ya que la tarea OBB exige localización más precisa.

* **Problemas atacados**: Caracterización exhaustiva del rendimiento de detectores bajo las condiciones específicas de imágenes aéreas.
* **Limitaciones de ese entonces**: No existían experimentos sistemáticos que analizaran el impacto de cada componente del detector en imágenes aéreas.
* **Soluciones alcanzadas**: Más de 70 configuraciones evaluadas, identificando las mejores prácticas para diseño de detectores orientados en imágenes aéreas.

##### 6.1.1 Mask Head vs. OBB Head
La cabeza Mask aborda la detección orientada como clasificación a nivel de píxel, convergiendo más fácilmente y logrando mejores resultados, pero con mayor costo computacional. La cabeza OBB trata el problema como regresión, siendo más rápida. En DOTA-v2.0, Mask R-CNN supera a Faster R-CNN H-OBB en 0.57 puntos de OBB mAP pero es 4 fps más lento.

##### 6.1.2 RoI Transformer vs. Deformable RoI Pooling
El RoI Transformer, diseñado específicamente para imágenes aéreas, supera al Deformable RoI Pooling general, validando que módulos de transformación geométrica diseñados a propósito para el dominio aéreo son superiores a módulos de propósito general.

##### 6.1.3 Excluding Small Instances
Los objetos extremadamente pequeños causan inestabilidad numérica durante el entrenamiento. Los experimentos muestran que filtrar instancias con área ≤ 80 y max(w,h) ≤ 10 tiene impacto mínimo en el mAP pero estabiliza el entrenamiento.

##### 6.1.4 Number of Proposals
El número óptimo de propuestas en DOTA es ~8,000, mucho mayor que los 300 óptimos de PASCAL VOC. Esto evidencia la diferencia fundamental entre la densidad de objetos en imágenes naturales y aéreas. Con 2,000 propuestas se obtiene buen balance velocidad-precisión.

##### 6.1.5 Data Augmentation
Los aumentos de datos multi-escala y rotacional mejoran sustancialmente el rendimiento: en DOTA-v1.5, de 65.03% (baseline) a 77.60% OBB mAP con todos los aumentos. Esto muestra que FPN y RoI Transformer no resuelven completamente el problema de variación de escala y orientación, y que el modelado geométrico con CNNs sigue siendo un problema abierto.

##### 6.1.6 Class-Wise Results
La comparación entre DOTA-v1.0 y v1.5 para la clase "small vehicles" muestra una caída de 25.4 puntos de AP al incluir objetos diminutos (de 77.45 a 52.05 con RoI Transformer). Las OBBs superan a las HBBs especialmente en objetos densos: Faster R-CNN OBB supera a Faster R-CNN en 8 puntos de AP en "large vehicles" en DOTA-v1.0.

##### 6.1.7 Visualization of the Results
La visualización muestra cuatro tipos de dificultades: (1) vehículos grandes densamente empaquetados (HBBs fallan, OBBs detectan bien), (2) instancias alargadas con gran AR (auto-similitud genera predicciones múltiples por instancia), (3) confusión entre categorías similares (puentes/aeropuertos/puertos), y (4) instancias extremadamente pequeñas con recall muy bajo.

* **Problemas atacados**: Identificación visual de los casos difíciles específicos de imágenes aéreas para guiar el desarrollo futuro.
* **Limitaciones de ese entonces**: Los detectores de imágenes naturales fallan sistemáticamente en estas cuatro categorías de dificultad.
* **Soluciones alcanzadas**: Catálogo visual de casos de fallo que sirve como guía de investigación futura.

#### 6.2 State-of-the-Art Results on DOTA-v1.0

Comparación con métodos del estado del arte en DOTA-v1.0: Faster R-CNN OBB + RoI Transformer logra 73.76% OBB mAP, superando a la mayoría excepto a Li et al. Con aumentos de datos, se alcanza 79.82%, superando a Li et al. (+3.46 puntos) y especialmente en categorías densas como "large vehicle" (+12.18 puntos). S2A-Net, con características espacialmente invariantes en un detector de una etapa, logra 79.42% con ResNet-50.

* **Problemas atacados**: Posicionamiento de los baselines respecto al estado del arte para validar la utilidad del dataset y la librería.
* **Limitaciones de ese entonces**: Métodos previos del estado del arte usaban múltiples escalas y augmentación de rotación, lo que hacía difícil separar la contribución del método de la del augmentación.
* **Soluciones alcanzadas**: Con las mismas configuraciones de augmentación, el método propuesto (RoI Transformer) supera al estado del arte previo en 3.46 puntos de OBB mAP.

#### 6.3 DOAI 2019 Challenge Results

DOTA-v1.5 fue usado para el desafío DOAI 2019 en CVPR (173 registros, 13 equipos en OBB, 22 en HBB). El método RoI Transformer con augmentación (modelo único) logró 77.60% y 78.88% en OBB y HBB respectivamente, siendo el mejor resultado reportado en la tarea OBB. Los equipos ganadores usaron ensambles de múltiples modelos, alcanzando hasta 78.34% en OBB.

* **Problemas atacados**: Validación del dataset como benchmark competitivo a nivel mundial.
* **Limitaciones de ese entonces**: Sin un desafío organizado, era difícil evaluar el estado real del arte en detección orientada.
* **Soluciones alcanzadas**: Atracción de 173 equipos internacionales, estableciendo DOTA como el benchmark estándar de facto para ODAI.

### 7. Conclusion

El artículo concluye reafirmando las tres contribuciones principales: (1) DOTA-v2.0 como el dataset más grande para Earth vision con OBBs; (2) una librería de código unificada para detección orientada; (3) baselines comprehensivos de más de 70 configuraciones. Los autores plantean continuar extendiendo el dataset, organizar más desafíos e integrar más algoritmos de detección orientada en la librería. Se enfatiza que DOTA puede complementar datasets de imágenes naturales para promover detectores de objetos universales.

* **Problemas atacados**: Síntesis de contribuciones y dirección futura para el campo de ODAI.
* **Limitaciones de ese entonces**: El modelado geométrico con CNNs y la detección de objetos diminutos siguen siendo problemas abiertos.
* **Soluciones alcanzadas**: Provisión de infraestructura completa (dataset, código, servidor de evaluación, desafíos) para catalizar el progreso en ODAI.
