# ArbiTrack: A Novel Multi-Object Tracking Framework for a Moving AAV to Detect and Track Arbitrarily Oriented Targets

- **Key**: Chen2025
- **Year**: 2025
- **Venue**: IEEE Transactions on Multimedia

## Resumen
Este artículo introduce "ArbiTrack", un marco novedoso de seguimiento de objetos múltiples (MOT) diseñado para vehículos aéreos autónomos (AAV o drones) en movimiento. Su propósito es detectar y rastrear eficientemente objetivos terrestres con orientaciones arbitrarias, distribuidos de forma densa en escenarios con fondos complejos. Para resolver las dificultades asociadas al movimiento irregular del dron, la escala pequeña de los objetos a gran altitud y las oclusiones temporales, ArbiTrack propone tres componentes core integrados en un paradigma de seguimiento por detección (TBD): (1) un detector de objetos orientados basado en "Oriented RepPoints", el cual incorpora un módulo de Agregación de Contexto Multiescala (MCA) que utiliza atención cruzada y pooling global para capturar características finas de objetivos pequeños; (2) un módulo de Conmutación Adaptativa de Movimiento (AMS) que selecciona y alterna dinámicamente entre un Filtro de Kalman (KF) para movimiento lineal (vuelo estable del dron) y un Filtro de Kalman Invariante/No lineal (Unscented Kalman Filter - UKF) para movimientos acelerados no lineales; y (3) un módulo de Memoria Evolutiva Espaciotemporal (SEM) basado en ConvGRU que modela la evolución espaciotemporal de los objetos y estima sus posiciones durante periodos de oclusión total o parcial. Los experimentos realizados en el dataset público UAVDT y su propio dataset anotado con orientaciones (OriDrone) demuestran que ArbiTrack supera notablemente a los rastreadores del estado del arte (como FairMOT y TrackFormer) en las métricas de precisión (MOTA), consistencia de trayectorias e ID switches (IDs).

## Secciones y Subsecciones

### I. Introduction
Establece los desafíos del seguimiento de objetos múltiples (MOT) desde plataformas móviles como vehículos aéreos autónomos (AAV).
* **Problemas atacados**: La pérdida de precisión de seguimiento provocada por el movimiento irregular y las maniobras tridimensionales del dron en vuelo, la deformación de las cajas delimitadoras de los vehículos al cambiar de ángulo visual, la alta presencia de objetos pequeños debido a la altitud de vuelo y el problema de oclusiones prolongadas en escenarios terrestres complejos.
* **Limitaciones de ese entonces**: Los algoritmos de MOT tradicionales asumen cámaras fijas u objetos simétricos en posición vertical (alineados con los ejes). Las cajas delimitadoras rectangulares estándar (AABB) capturan demasiado fondo irrelevante en objetos rotados, lo que causa imprecisión. Además, los filtros de movimiento lineal estándar (KF) fallan al modelar la compleja cinemática resultante de superponer la velocidad del dron y la del vehículo terrestre. Los filtros relacionales previos asumen un número fijo de objetos entre fotogramas sucesivos, colapsando ante oclusiones del mundo real.
* **Soluciones alcanzadas**: Se propone el marco ArbiTrack. Este utiliza Oriented RepPoints para extraer la orientación de las cajas y puntos adaptativos geométricos. Se diseña un módulo MCA para recuperar características de objetos de tamaño reducido y un estimador AMS basado en retroalimentación del estado de aceleración para alternar esquemas de filtrado. Se crea un almacenamiento SEM para gestionar la permanencia de los objetos.

### II. Related Work
Explora las contribuciones previas en la detección, seguimiento y asociación de datos en flujos de video de drones.

#### A. Tracking-by-Detection
* **Problemas atacados**: La evolución de los rastreadores desde esquemas de dos etapas (detección y asociación separadas) a esquemas de una sola etapa (detección y extracción de ReID conjuntas).
* **Limitaciones de ese entonces**: Los métodos clásicos de dos etapas (como SORT y DeepSORT) sufren de frecuentes ID switches en escenarios dinámicos. Los métodos rápidos de una sola etapa (JDE, FairMOT, CenterTrack) están optimizados para cámaras de vigilancia estáticas terrestres y fallan cuando se aplican a videos aéreos de alta variabilidad debido a la densa distribución de vehículos y la interferencia de fondos complejos. Los métodos térmicos infrarrojos (TIR) ayudan en clima adverso pero sufren deformación severa y falta de texturas visuales.
* **Soluciones alcanzadas**: Se introduce un detector de objetos orientados que reemplaza a los detectores AABB tradicionales. Este proporciona datos de rotación angular del bounding box como una señal geométrica adicional que asiste directamente al proceso de asociación de trayectorias.

#### B. Data Association
* **Problemas atacados**: La formulación de métricas de afinidad basadas en movimiento y apariencia para emparejar detecciones con trayectorias.
* **Limitaciones de ese entonces**: El movimiento observado desde un dron en movimiento es altamente no lineal. Las aproximaciones basadas en redes neuronales para predecir movimiento añaden una carga computacional prohibitiva en entornos integrados. Por otro lado, las firmas de apariencia son inestables por efectos de desenfoque y oclusión. Los rastreadores basados en Transformers (como TransTrack y TrackFormer) propagan consultas entre fotogramas adyacentes pero no consideran la historia a largo plazo, limitando su capacidad para reconectar trayectorias tras oclusiones extensas.
* **Soluciones encargadas**: El módulo AMS calcula la cinemática sin agregar redes pesadas de aprendizaje profundo. Adicionalmente, el módulo SEM basado en ConvGRU preserva y propaga la memoria espaciotemporal del objeto a lo largo de múltiples pasos de tiempo para guiar la localización cuando el detector pierde de vista al objetivo.

### III. Methodology
Presenta los detalles matemáticos y arquitectónicos del marco ArbiTrack.

#### A. Overall Framework
* **Problemas atacados**: Diseñar un flujo de red unificado que reciba secuencias de video a la entrada y devuelva trayectorias vehiculares orientadas en tiempo real.
* **Limitaciones de ese entonces**: La pérdida de información geométrica fina y el desalineamiento espacial al usar capas FPN estándar.
* **Soluciones alcanzadas**: El video ingresa a un backbone extractor de características. FPN se sustituye por el módulo MCA. Tras la cabeza de detección orientada, se genera un mapa de características ID y coordenadas angulares de Oriented RepPoints. Los datos de movimiento se estiman en el bloque AMS y, junto con los mapas visuales, alimentan el bloque SEM para el rastreo final.

#### B. Multi-Scale Context Aggregation
* **Problemas atacados**: Conservar y realzar la representación de características de objetos pequeños en mapas de alta resolución.
* **Limitaciones de ese entonces**: Las redes FPN tradicionales suelen asignar erróneamente objetos de tamaños similares a diferentes niveles de la pirámide de características y consumen mucho tiempo de cómputo al procesar cada capa de predicción de forma separada.
* **Soluciones alcanzadas**: Se propone el módulo MCA. Este toma los mapas de características de múltiples etapas de la red (C2, C3, C4, C5). Usa mecanismos de atención cruzada (Cross-Attention) para capturar contexto local entre capas contiguas y pooling de promedio global para el contexto general de C5. Las salidas se fusionan mediante pesos de aprendizaje ponderados acotados (fusion factors), generando un mapa único y robusto para la cabeza de detección, evitando la redundancia de predecir en múltiples escalas independientes.

#### C. Adaptive Motion Switching Module
* **Problemas atacados**: Modelar con precisión la cinemática no lineal de los vehículos en relación con el movimiento acelerado y giros rápidos del dron.
* **Limitaciones de ese entonces**: El Filtro de Kalman convencional (KF) solo es óptimo para dinámicas lineales de velocidad constante. Si el dron frena, acelera o gira bruscamente, la predicción lineal falla completamente, provocando la pérdida del track.
* **Soluciones alcanzadas**: Se introduce un clasificador de modo de movimiento (normal y anormal) controlado por un umbral de aceleración $\tau_a$. En el modo normal (vuelo lineal/estable), el sistema utiliza Kalman Filtering (KF) por su bajo costo computacional. En el modo anormal (maniobras rápidas o aceleración del dron), el sistema conmuta automáticamente a Unscented Kalman Filtering (UKF), modelando de forma precisa la dinámica no lineal y manteniendo la coherencia espacial del track.

#### D. Spatio-Temporal Evolutionary Memory
* **Problemas atacados**: Recuperar las identidades de vehículos que sufren oclusiones completas prolongadas causadas por árboles, puentes o edificios en la escena aérea.
* **Limitaciones de ese entonces**: Cuando un objeto desaparece, los rastreadores convencionales destruyen la trayectoria o generan un nuevo identificador al reaparecer el vehículo, lo que incrementa el ID switching y fragmenta la trayectoria.
* **Soluciones alcanzadas**: Se implementa un módulo SEM basado en ConvGRU que reemplaza los vectores 1D tradicionales por mapas de características 2D para mantener la coherencia espacial del objeto. Acepta el mapa de características visuales IDFt, el estado anterior $S_{t-1}$ y la matriz cinemática $M_t$ provista por el módulo AMS para estimar la posición oculta del vehículo. Si el detector no registra al objeto, el sistema guarda la predicción de ConvGRU en la memoria de estado. Al reaparecer el vehículo en un radio $k$ (calculado por las dimensiones del cuadro), se calcula una afinidad greedy combinando la distancia del centro y la métrica de IoU Completo (CIoU) para reconectar el track original de forma exitosa.

### IV. Experiments
Describe la fase de experimentación, bases de datos empleadas y análisis comparativos.

#### A. Datasets and Metrics
* **Problemas atacados**: Validar empíricamente la robustez de ArbiTrack en diferentes condiciones de vuelo y tráfico, utilizando métricas estandarizadas de MOT.
* **Limitaciones de ese entonces**: Falta de conjuntos de datos de seguimiento desde drones que incluyan anotaciones angulares (orientación de cajas) para vehículos.
* **Soluciones alcanzadas**: Se presenta el dataset "OriDrone" capturado con un dron DJI Phantom 4 Pro a resolución 1280x720. Contiene 40 secuencias divididas en entrenamiento y prueba, anotadas con ID, clase y cajas orientadas. También se utiliza el dataset público UAVDT (50 secuencias, resolución 1080x540) adaptando sus etiquetas. Las métricas evaluadas son MOTA, MOTP y ID switches (IDs).

#### B. Implementation Details
* **Problemas atacados**: Configurar y entrenar el modelo de forma óptima para alcanzar convergencia.
* **Limitaciones de ese entonces**: El entrenamiento de redes híbridas que acoplan detección de puntos RepPoints y memoria recurrente puede presentar inestabilidad numérica.
* **Soluciones alcanzadas**: Se configura el ConvGRU con dimensiones de característica de 256 y kernels de convolución de 7x7. Se utiliza optimizador Adam por 30 épocas en una GPU RTX 3090, aplicando un decaimiento progresivo del factor de aprendizaje en las épocas 10 y 20.

#### C. Comparison With State-of-the-Arts
* **Problemas atacados**: Comparar el desempeño cuantitativo de ArbiTrack frente a rastreadores consolidados de la industria.
* **Limitaciones de ese entonces**: La mayoría de los métodos de MOT no soportan cajas delimitadoras rotadas, requiriendo adaptar las anotaciones a rectángulos verticales (AABB) para una comparación justa.
* **Soluciones alcanzadas**: En la base OriDrone, ArbiTrack logró 44.6% en MOTA y 56.9% en IDF1, superando a FairMOT (42.2% MOTA) y TrackFormer (37.9% MOTA) con una reducción drástica de IDs. En UAVDT, alcanzó 47.6% en MOTA y 67.4% en IDF1, superando ampliamente a FairMOT y consolidando la efectividad de las cajas orientadas.

#### D. Ablation Study
* **Problemas atacados**: Aislar la ganancia en exactitud y velocidad (FPS) aportada por cada módulo de forma individual.
* **Limitaciones de ese entonces**: UKF es altamente preciso pero lento computacionalmente, mientras que KF es rápido pero inestable bajo aceleraciones.
* **Soluciones alcanzadas**: El análisis de AMS demostró que KF solo provee 12 FPS pero genera 1340 IDs, mientras que UKF puro reduce los IDs a 563 pero a costa de caer a 5 FPS. El AMS híbrido balancea ambos mundos logrando 10 FPS y 786 IDs. Por su parte, la inclusión progresiva de MCA, AMS y SEM sobre el baseline SORT incrementó secuencialmente la métrica MOTA de 38.4% a 44.6% e IDF1 de 44.6% a 56.9%.

#### E. Visualization
* **Problemas atacados**: Analizar cualitativamente el comportamiento del rastreador bajo giros repentinos del dron, oclusiones temporales, cambios de altitud y fondos complejos.
* **Limitaciones de ese entonces**: Falta de análisis visual directo para identificar fallas en el anclaje de ID de otros modelos comerciales.
* **Soluciones alcanzadas**: La visualización demostró que FairMOT pierde vehículos durante giros bruscos del AAV o en intersecciones congestionadas con objetos pequeños. Por el contrario, ArbiTrack mantiene los IDs estables y recupera vehículos tras oclusiones de hasta 2.5 segundos (como el carro ID 58 en la base OriDrone) gracias a la persistencia espaciotemporal del módulo SEM.

### V. Conclusion
Sintetiza las conclusiones y presenta las líneas de investigación futuras.
* **Problemas atacados**: Resumir la efectividad del framework ArbiTrack y proyectar mejoras tecnológicas.
* **Limitaciones de ese entonces**: La degradación de la calidad de detección y seguimiento durante la noche o en condiciones de oscuridad extrema utilizando únicamente cámaras visibles de AAV.
* **Soluciones alcanzadas**: Se valida la robustez de ArbiTrack para escenarios complejos de luz diurna. Como trabajo futuro, se propone explorar la fusión de sensores de video visibles e infrarrojos para realizar MOT nocturno eficiente.
