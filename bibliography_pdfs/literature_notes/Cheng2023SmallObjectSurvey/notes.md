# Towards Large-Scale Small Object Detection: Survey and Benchmarks

- **Key**: Cheng2023SmallObjectSurvey
- **Year**: 2023
- **Venue**: arXiv:2207.14096v4 [cs.CV] / IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

## Resumen

Este artículo realiza una revisión exhaustiva del campo de detección de objetos pequeños (Small Object Detection, SOD) en la era del aprendizaje profundo, abordando tanto los algoritmos como los conjuntos de datos disponibles. Los autores argumentan que, a pesar del notable progreso en detección de objetos genérica impulsado por las redes neuronales convolucionales profundas, el SOD permanece rezagado: el mejor detector de su época (DyHead) solo alcanza un mAP de 28.3% en objetos pequeños en COCO, frente al 57.5% en objetos grandes. Identifican dos causas principales: (1) la dificultad intrínseca de aprender representaciones a partir de información limitada y distorsionada de los objetos pequeños, y (2) la escasez de conjuntos de datos a gran escala para SOD. Como respuesta, construyen dos benchmarks de gran escala denominados SODA (Small Object Detection dAtasets): SODA-D para escenarios de conducción (24,828 imágenes, 278,433 instancias, 9 categorías con anotaciones horizontales) y SODA-A para escenas aéreas (2,513 imágenes de alta resolución, 872,069 instancias con anotaciones de caja orientada). Organizan los métodos SOD en seis categorías taxonómicas y evalúan 12 detectores representativos sobre SODA-D y 9 sobre SODA-A, identificando importantes brechas de rendimiento y direcciones futuras de investigación.

## Secciones y Subsecciones

### 1. Introduction
La introducción motiva el trabajo explicando la brecha de rendimiento persistente entre la detección de objetos genérica y la detección de objetos pequeños. Se señala que incluso los detectores de punta muestran diferencias drásticas de rendimiento entre objetos pequeños, medianos y grandes. Los autores posicionan su contribución como la primera revisión comprehensiva dedicada exclusivamente al SOD a través de múltiples dominios, diferenciándose de surveys previos que o bien se enfocaban en detección genérica o cubrían SOD de manera parcial. Se presentan las tres contribuciones principales: la revisión sistemática y taxonomía de seis categorías, la construcción de los benchmarks SODA-D y SODA-A, y la evaluación exhaustiva de métodos representativos sobre dichos benchmarks.
* **Problemas atacados**: La falta de una revisión sistemática y comprensiva del SOD, y la ausencia de benchmarks a gran escala dedicados específicamente a la detección de objetos pequeños multiclase.
* **Limitaciones de ese entonces**: Los surveys existentes sobre SOD eran parciales, cubrían dominios limitados, o mezclaban métodos de detección genérica con métodos SOD sin rigor. Los benchmarks existentes (SOD, TinyPerson) eran de pequeña escala o de categoría única.
* **Soluciones alcanzadas**: Se propone la taxonomía más comprehensiva hasta la fecha del SOD (seis categorías) y se anuncian los primeros benchmarks a gran escala multiclase dedicados al SOD.

#### 1.1 Comparisons with Previous Reviews
Esta subsección compara el survey propuesto con revisiones previas de detección de objetos, destacando que la mayoría de surveys anteriores se concentran en detección genérica o tareas específicas como detección de peatones o señales de tráfico. Los pocos surveys previos sobre SOD no logran cobertura comprehensiva o mezclan indistintamente métodos SOD con genéricos. El artículo propone una taxonomía propia de seis categorías y benchmarks a gran escala que ningún survey previo incluía.
* **Problemas atacados**: La falta de un marco taxonómico claro y comprehensivo que organice los métodos SOD de forma rigurosa.
* **Limitaciones de ese entonces**: Los surveys SOD previos cubrían solo partes del campo, carecían de benchmarks propios y usaban datasets genéricos dominados por objetos medianos y grandes para sus evaluaciones.
* **Soluciones alcanzadas**: Se ofrece una taxonomía de seis grupos que organiza cientos de trabajos de literatura SOD de manera sistemática y con análisis comparativo profundo.

#### 1.2 Scope
Define el alcance del survey: se enfoca exclusivamente en métodos SOD basados en aprendizaje profundo (desde 2012 en adelante), dejando fuera los métodos con características artesanales que tienen capacidad limitada de variación de escala. Se justifica esta delimitación por el enorme salto de rendimiento que trajo el aprendizaje profundo y su relevancia para el estado del arte actual.
* **Problemas atacados**: Necesidad de delimitar claramente qué se revisa, dado que la detección basada en características artesanales es históricamente obsoleta para SOD.
* **Limitaciones de ese entonces**: Los métodos basados en características artesanales fallaban catastróficamente en objetos pequeños por su capacidad limitada ante variaciones de escala.
* **Soluciones alcanzadas**: Delimita el survey a métodos deep learning, asegurando que la revisión sea relevante y actualizada para el estado del arte.

### 2. Review on Small Object Detection

#### 2.1 Problem Definition
Define formalmente qué es un objeto pequeño, siguiendo criterios de área o longitud: en COCO, un objeto con área ≤ 1024 píxeles se considera pequeño. Se aclara que el término "tiny" y "small" son usados intercambiablemente en la literatura original, y se indica que el paper define su propio criterio de "Small" para el benchmark SODA.
* **Problemas atacados**: La falta de una definición estándar y unificada del término "objeto pequeño" en la literatura.
* **Limitaciones de ese entonces**: Distintos trabajos usaban diferentes umbrales de área o longitud, haciendo difícil la comparación directa de resultados.
* **Soluciones alcanzadas**: Adopta una definición consistente con trabajos previos (COCO) y se propone una clasificación interna propia para el benchmark SODA.

#### 2.2 Main Challenges
Identifica y analiza los cuatro retos principales del SOD: (1) pérdida de información debida al subsampling en redes profundas que elimina señales de objetos pequeños; (2) representación de características ruidosa por la baja resolución y contaminación de fondo; (3) baja tolerancia a la perturbación de bounding box, donde una desviación de 6 píxeles reduce el IoU de 100% a 32.5% en objetos pequeños (versus 71.8% en objetos grandes); (4) muestras insuficientes para entrenamiento, pues los esquemas de asignación basados en solapamiento fallan con instancias pequeñas.
* **Problemas atacados**: Entender y caracterizar sistemáticamente por qué el SOD es sustancialmente más difícil que la detección de objetos normales.
* **Limitaciones de ese entonces**: Los detectores genéricos no están diseñados para manejar la extrema sensibilidad de los objetos pequeños a perturbaciones de caja y la pérdida de información en capas profundas.
* **Soluciones alcanzadas**: Proporciona un análisis cuantitativo claro de la baja tolerancia a perturbaciones y sienta las bases motivacionales para las seis categorías de soluciones revisadas.

#### 2.3 Review of Small Object Detection Algorithms
Revisión sistemática organizada en seis grupos de métodos. Se comienza describiendo el contexto de los detectores genéricos (two-stage, one-stage, anchor-free y query-based) antes de detallar los métodos SOD especializados.
* **Problemas atacados**: Organizar y sintetizar cientos de trabajos de la literatura SOD en una taxonomía coherente y manejable.
* **Limitaciones de ese entonces**: La literatura SOD estaba dispersa sin una taxonomía clara que permitiera entender las relaciones entre los distintos enfoques.
* **Soluciones alcanzadas**: Propone seis categorías claras que cubren todo el espectro de enfoques SOD, con análisis de pros y contras de cada grupo.

##### 2.3.1 Sample-oriented Methods
Aborda los métodos que atacan el problema de muestras insuficientes para entrenamiento. Se dividen en dos sub-grupos: estrategias de aumento de datos (copiar y pegar objetos pequeños con transformaciones, uso de segmentación para guiar el pegado, generación sintética mediante DS-GAN) y estrategias de asignación óptima de etiquetas (S3FD con compensación de escala, EMO para anclas, DotD basado en distancia euclidiana normalizada, RFLA con campo receptivo gaussiano). Se concluye que los métodos de augmentación sufren de mejora inconsistente y baja transferibilidad, mientras que los esquemas de asignación óptima aún luchan con instancias extremadamente pequeñas.
* **Problemas atacados**: La escasez de muestras positivas durante el entrenamiento, pues los objetos pequeños tienen solapamientos mínimos con los anchors.
* **Limitaciones de ese entonces**: Los esquemas de asignación basados en IoU estándar asignaban muy pocas muestras positivas a instancias pequeñas, dejando esas regiones sub-optimizadas.
* **Soluciones alcanzadas**: Diversas estrategias de augmentación y de asignación de etiquetas que incrementan las muestras positivas disponibles para objetos pequeños.

##### 2.3.2 Scale-aware Methods
Revisa los métodos que abordan la variación de escala mediante arquitecturas multi-rama o estrategias de entrenamiento a múltiples escalas. Se describen los detectores específicos por escala (FPN, MS-CNN, YOLOv3, TridentNet, SNIP, Sniper, AutoFocus) y los métodos de fusión de características jerárquicas (PANet, StairNet, IPG-Net, SSPNet). Se analiza la tensión entre la ventaja de asignar objetos a niveles óptimos de la pirámide y el problema de que la información de un solo nivel puede ser insuficiente.
* **Problemas atacados**: La variación de escala severa en imágenes de tráfico y teledetección que causa dificultades dispares para detectores de escala única.
* **Limitaciones de ese entonces**: Los detectores tempranos usaban solo características de alto nivel para reconocimiento, perdiendo información fina de escala necesaria para objetos pequeños.
* **Soluciones alcanzadas**: Arquitecturas piramidales (FPN y variantes) y esquemas de fusión de características que integran múltiples niveles para representaciones más ricas de objetos pequeños.

##### 2.3.3 Attention-based Methods
Revisa métodos que aplican mecanismos de atención visual para destacar regiones de objetos pequeños y suprimir el fondo ruidoso. Ejemplos incluyen SCRDet (atención de píxeles y canales supervisada), FBR-Net (atención basada en nivel de pirámide), KB-RANN (redes de atención a largo y corto plazo) y MSCCA (bloques ECA ligeros). Se señala que los métodos de atención tienen diseños de inserción flexible, pero incurren en sobrecarga computacional por las operaciones de correlación y carecen de supervisión directa.
* **Problemas atacados**: La tendencia de las características de objetos pequeños a ser dominadas por el fondo y los patrones ruidosos, dificultando la extracción de información discriminativa.
* **Limitaciones de ese entonces**: Los mecanismos de atención genéricos no estaban diseñados específicamente para destacar los débiles patrones de objetos pequeños frente a fondos dominantes.
* **Soluciones alcanzadas**: Módulos de atención adaptados al SOD que destacan regiones de objetos pequeños y suprimen el ruido, mejorando la representación discriminativa.

##### 2.3.4 Feature-imitation Methods
Revisa métodos que buscan enriquecer las representaciones de objetos pequeños imitando las de objetos más grandes. Se dividen en métodos de aprendizaje por similaridad (SML, LPR que fuerzan a las características de objetos pequeños a aproximarse a las de grandes) y frameworks basados en super-resolución (uso de deconvolución, sub-pixel convolution, GAN como PerceptualGAN, MTGAN). Se concluye que estos métodos deben evitar el colapso de características y que los métodos GAN pueden fabricar texturas falsas perjudiciales.
* **Problemas atacados**: La baja calidad de representación de características de objetos pequeños derivada de su limitada información visual, especialmente en instancias extremadamente pequeñas.
* **Limitaciones de ese entonces**: Las características de objetos pequeños son inherentemente de baja calidad y difíciles de enriquecer sin recurrir a información de instancias más grandes.
* **Soluciones alcanzadas**: Métodos que transfieren información de representación de objetos grandes a pequeños mediante restricciones de similaridad o reconstrucción super-resolución.

##### 2.3.5 Context-modeling Methods
Revisa métodos que explotan la información de contexto para mejorar la detección de objetos pequeños. Ejemplos incluyen PyramidBox, SINet, IONet y R2-CNN. Se señala la limitación de que los mecanismos actuales determinan las regiones de contexto de manera heurística sin garantía de interpretabilidad.
* **Problemas atacados**: El hecho de que los objetos pequeños tienen información visual insuficiente por sí mismos, siendo el contexto circundante crucial para su correcta identificación.
* **Limitaciones de ese entonces**: Los detectores genéricos solo analizaban la región inmediata del objeto, ignorando el contexto circundante que puede ser más informativo que el propio objeto pequeño.
* **Soluciones alcanzadas**: Módulos de modelado de contexto que integran información de regiones vecinas y globales para generar representaciones más de objetos pequeños.

##### 2.3.6 Focus-and-detect Methods
Revisa métodos que primero localizan las regiones de interés en imágenes de alta resolución y luego aplican detección focalizada, evitando el procesamiento redundante de regiones vacías. Ejemplos: ClusDet, DMNet, CRENet y F&S. La limitación principal es la necesidad de anotaciones adicionales o arquitecturas auxiliares complejas para determinar "dónde enfocar".
* **Problemas atacados**: La ineficiencia del procesamiento de ventana deslizante uniforme en imágenes de alta resolución donde los objetos pequeños se distribuyen de forma no uniforme y dispersa.
* **Limitaciones de ese entonces**: Los detectores genéricos desperdiciaban cómputo en parches vacíos y procesaban objetos pequeños a resoluciones subóptimas.
* **Soluciones alcanzadas**: Pipelines de dos etapas que filtran regiones vacías y aplican zoom-in adaptativo a regiones de interés, procesando objetos pequeños a resoluciones más altas.

### 3. Review of Datasets for Small Object Detection

#### 3.1 Datasets for Small Object Detection
Revisión de datasets existentes que contienen objetos pequeños, abarcando detección de caras (WiderFace), peatones (TinyPerson, EuroCity Persons), señales de tráfico (TT100K), drones (VisDrone) y teledetección (DOTA, AI-TOD). Se destaca que la mayoría de estos datasets son de categoría única o no están diseñados específicamente para SOD multiclase. Se incluye una tabla comparativa de datasets en el paper.
* **Problemas atacados**: La carencia de un conjunto de datos a gran escala multiclase diseñado específicamente para benchmarking de SOD.
* **Limitaciones de ese entonces**: Los datasets existentes eran de categoría única (detección de caras/peatones) o, cuando multiclase, tenían objetos pequeños concentrados en pocas categorías o carecían de escala suficiente.
* **Soluciones alcanzadas**: Enfoque en el diseño y justificación del conjunto de datos SODA para llenar este vacío en escenarios tanto terrestres como aéreos.

#### 3.2 Evaluation Metrics
Describe las métricas de evaluación estándar: Average Precision (AP) y COCO AP. Explica los conceptos de TP, FP, FN y el AP de multi-IoU.
* **Problemas atacados**: La necesidad de métricas de evaluación que capturen adecuadamente el rendimiento en objetos de diversas escalas.
* **Limitaciones de ese entonces**: La métrica AP de umbral único (IoU=0.5) no incentivaba la localización precisa; para objetos pequeños, pequeñas desviaciones causan caídas dramáticas de IoU.
* **Soluciones alcanzadas**: Adopción de métricas específicas como APeS, APrS, APgS y APN para cuantificar granularmente el rendimiento en objetos pequeños y normales.

### 4. Benchmarks (SODA)

#### 4.1 Data Acquisition and Annotation
Describe el proceso de construcción de SODA-D y SODA-A. SODA-D tiene 24,828 imágenes de conducción anotadas con 278,433 cajas horizontales. SODA-A tiene 2,513 imágenes aéreas de alta resolución anotadas con 872,069 cajas orientadas. Define el umbral para "Small" en S ≤ 1024 píxeles cuadrados.
* **Problemas atacados**: La necesidad de criterios de anotación rigurosos y consistentes para construir benchmarks auténticamente dedicados a SOD.
* **Limitaciones de ese entonces**: Los datasets existentes no tenían criterios explícitos para seleccionar y anotar objetos pequeños, resultando en una mezcla de instancias de diferentes tamaños.
* **Soluciones alcanzadas**: Un protocolo de anotación detallado con definición cuantitativa de "objeto valioso", sub-clasificación de pequeñez y regiones de "ignore" para instancias ambiguas.

#### 4.2 Statistical Analysis
Análisis estadístico que demuestra que SODA-D y SODA-A superan ampliamente a datasets existentes en cantidad de instancias extremadamente pequeñas y densidad de objetos por imagen.
* **Problemas atacados**: Demostrar cuantitativamente que SODA representa un avance significativo sobre benchmarks previos en términos de escala, densidad y especificidad para SOD.
* **Limitaciones de ese entonces**: Los datasets previos tenían pocas instancias extremadamente pequeñas o las concentraban en pocas categorías, no representando bien la diversidad real del SOD.
* **Soluciones alcanzadas**: SODA proporciona cientos de miles de instancias extremadamente pequeñas distribuidas en 9 categorías balanceadas, cubriendo tanto escenarios de conducción como aéreos.

#### 4.3 Comparisons with Previous Benchmarks
Compara SODA-D con MVD y SODA-A con AI-TOD y DOTA, justificando los avances en resolución, densidad, balance de clases e inclusión de anotaciones rotadas.
* **Problemas atacados**: Posicionar SODA como complemento necesario al ecosistema de datasets existente para SOD.
* **Limitaciones de ese entonces**: AI-TOD tenía desbalance de clases y resolución baja; DOTA tenía objetos pequeños concentrados en pocas categorías; ninguno proveía orientaciones arbitrarias con alta densidad y balance de clases simultáneamente.
* **Soluciones alcanzadas**: SODA-A combina alta resolución, orientaciones libres, balance de clases y volumen masivo de instancias pequeñas en un único benchmark.

### 5. Experiments

#### 5.1 Evaluation Protocol
Adopta el protocolo COCO de AP promediada sobre 10 umbrales de IoU (0.5 a 0.95). Define métricas específicas por tamaño: APeS, APrS, APgS y APN.
* **Problemas atacados**: Asegurar una evaluación justa y representativa del rendimiento real en objetos pequeños de diversas sub-escalas.
* **Limitaciones de ese entonces**: Las métricas genéricas de AP no desglosaban el rendimiento por sub-rangos de tamaño dentro de la categoría "small", ocultando diferencias importantes.
* **Soluciones alcanzadas**: Protocolo de evaluación multi-IoU con desglose por sub-categorías de tamaño, permitiendo análisis más granular del rendimiento en SOD.

#### 5.2 Implementation Details
Detalla la configuración experimental: backbone ResNet-50 por defecto, 4 GPUs NVIDIA RTX 3090, batch size de 8 (SODA-D) y 4 (SODA-A). Se usan 12 métodos representativos para SODA-D y 9 para SODA-A.
* **Problemas atacados**: Garantizar reproducibilidad y comparación justa entre todos los métodos evaluados.
* **Limitaciones de ese entonces**: Los benchmarks previos usaban configuraciones experimentales heterogéneas que dificultaban la comparación directa.
* **Soluciones alcanzadas**: Marco experimental unificado sobre mmdetection/mmrotate con configuración estándar para todos los métodos.

#### 5.3 Results Analysis on SODA-D
Analiza el desempeño de detectores como Cascade RCNN, FCOS, CenterNet, CornerNet, YOLOX y Deformable DETR en SODA-D, encontrando brechas importantes especialmente en objetos extremadamente pequeños.
* **Problemas atacados**: Entender el rendimiento real de los detectores de última generación en un benchmark auténticamente orientado a SOD en conducción.
* **Limitaciones de ese entonces**: No existía un benchmark dedicado multiclase para SOD en conducción que permitiera identificar las fortalezas y debilidades de cada paradigma detector.
* **Soluciones alcanzadas**: SODA-D revela brechas específicas: los detectores de dos etapas superan a los de una etapa en objetos extremadamente pequeños, y los backbones de tipo transformer son prometedores para SOD.

#### 5.4 Results Analysis on SODA-A
Analiza el rendimiento en SODA-A de detectores de OBB como RoI Transformer, Oriented RCNN y Oriented RepPoints.
* **Problemas atacados**: Entender el rendimiento de métodos de detección orientada (OBB) en el contexto específico de objetos pequeños aéreos.
* **Limitaciones de ese entonces**: Los benchmarks OBB existentes (DOTA) no estaban optimizados para SOD, mezclando objetos de múltiples escalas y sesgando el análisis.
* **Soluciones alcanzadas**: SODA-A permite identificar que propuestas de alta calidad son fundamentales para SOD aéreo y que la representación puntual es desventajosa para objetos pequeños de alta relación de aspecto.

### 6. Conclusion and Outlook
Resume las contribuciones del trabajo y propone cuatro direcciones futuras de investigación (extractores eficientes de características, FPNs optimizadas para SOD, asignación óptima de etiquetas para objetos pequeños, y métricas de evaluación más flexibles).
* **Problemas atacados**: Identificar las limitaciones actuales de los métodos SOD estado del arte y señalar las brechas más urgentes a resolver.
* **Limitaciones de ese entonces**: Los detectores más potentes aún tienen un rendimiento muy bajo en objetos extremadamente pequeños, y las métricas estándar pueden no capturar bien el progreso real en SOD.
* **Soluciones alcanzadas**: Se provee una agenda de investigación concreta con cuatro líneas de trabajo que el campo necesita abordar para impulsar el SOD hacia niveles de rendimiento comparables con la detección de objetos normales.
