# Vision Technologies with Applications in Traffic Surveillance Systems: A Holistic Survey

- **Key**: Zhou2025TrafficSurveillanceSurvey
- **Year**: 2025
- **Venue**: ACM Computing Surveys (CSUR)

## Resumen

Este artículo presenta una revisión holística y sistemática de las tecnologías de visión computacional aplicadas a los Sistemas de Vigilancia de Tráfico (Traffic Surveillance Systems, TSS) en el contexto de las ciudades inteligentes. Propone un marco de análisis unificado que conecta las tareas de percepción de bajo nivel (detección 2D/3D, clasificación fina de modelos y reidentificación de vehículos, y seguimiento de objetos individuales/múltiples) con las tareas de percepción de alto nivel (estimación de parámetros de tráfico, detección de anomalías y comprensión del comportamiento de vehículos y usuarios vulnerables). El artículo identifica cinco limitaciones fundamentales de los sistemas actuales (degradación de datos visuales, restricciones del aprendizaje supervisado clásico, brecha semántica, límites de cobertura física y demandas computacionales) y evalúa críticamente las soluciones y tendencias emergentes. Finalmente, analiza el impacto de los modelos fundacionales (LLMs, LVMs, VLMs y modelos de mundo físico FWMs como Sora) como la próxima frontera para revolucionar la resiliencia y el razonamiento de los sistemas de tránsito.

## Secciones y Subsecciones

### 1. Introducción
Los Sistemas de Vigilancia de Tráfico (TSS) son esenciales para los Sistemas Inteligentes de Transporte (ITS). Las cámaras de vigilancia son el sensor predominante debido a que proveen información semántica y visual de alta resolución a bajo costo. El avance de técnicas de aprendizaje profundo ha superado la dependencia de descriptores manuales tradicionales.
* **Problemas atacados**: Fragmentación en la literatura previa que aísla las tareas de bajo nivel (detección) de las de alto nivel (comportamiento y anomalías), omitiendo tecnologías emergentes como los modelos fundacionales.
* **Limitaciones de ese entonces**: Las revisiones previas carecen de un análisis técnico comparativo profundo de los trade-offs metodológicos y no analizan el potencial integrador de los Large Vision-Language Models (VLMs).
* **Soluciones alcanzadas**: Creación de un marco unificado de percepción de dos niveles (bajo y alto nivel) y desarrollo de un roadmap estructurado integrando modelos fundacionales y aprendizaje eficiente.

### 2. Overview
Describe la estructura general del artículo mapeando las secciones del ciclo de percepción.
* **Problemas atacados**: Dificultades para conceptualizar el flujo completo de datos desde la captura de píxeles hasta la toma de decisiones semánticas de alto nivel.
* **Limitaciones de ese entonces**: Falta de diagramas de flujo unificados que integren tareas dispares como calibración de cámaras, seguimiento e intención de peatones.
* **Soluciones alcanzadas**: Diseño de una arquitectura conceptual integrada que conecta entradas visuales con salidas predictivas complejas (trayectorias, alertas, preguntas y respuestas visuales).

### 3. Low-Level Traffic Perception Tasks

#### 3.1 Detection
Identificación y localización espacial de participantes del tráfico a través de cajas delimitadoras.

##### 3.1.1 2D Detection
Clasificación y comparación de detectores de dos etapas y de una etapa.
* **Problemas atacados**: Necesidad de balancear la precisión de detección de objetos pequeños en entornos urbanos congestionados con los tiempos de respuesta exigidos en tiempo real.
* **Limitaciones de ese entonces**: Los detectores de dos etapas sufren de alta latencia inherente. Los detectores de una sola etapa basados en anclas fallan ante orientaciones de objetos inusuales o densidades compactas.
* **Soluciones alcanzadas**: Evolución hacia detectores anchor-free de una etapa y adaptaciones de Transformers en tiempo real (RT-DETR) que logran optimizar el equilibrio entre velocidad y precisión.

##### 3.1.2 3D Detection
Generación de cajas orientadas tridimensionales a partir de imágenes monoculares de infraestructura vial.
* **Problemas atacados**: Inferencia imprecisa de la profundidad espacial y dimensiones reales de los vehículos desde vistas fijas elevadas.
* **Limitaciones de ese entonces**: Los métodos basados en restricciones geométricas puras son sumamente sensibles a errores de calibración y movimientos físicos de la cámara.
* **Soluciones alcanzadas**: Adopción de métodos de estimación directa mediante aprendizaje profundo extremo a extremo que estiman altura y profundidad normalizada para contrarrestar la incertidumbre espacial.

#### 3.2 Classification
Identificación fina de atributos específicos de los vehículos.

##### 3.2.1 Vehicle model recognition (VLR)
Reconocimiento de marcas, logotipos y modelos específicos de vehículos.
* **Problemas atacados**: Variaciones extremas de apariencia visual dentro de la misma categoría de vehículo y oclusión de logotipos viales.
* **Limitaciones de ese entonces**: Los descriptores tradicionales colapsan ante ángulos oblicuos o variaciones de iluminación nocturna.
* **Soluciones alcanzadas**: Modelos basados en Transformers de visión (Swin-T) y cabezales híbridos multicapa que reconocen con alta precisión modelos y marcas.

##### 3.2.2 Vehicle re-identification (Re-ID)
Asociación y búsqueda de un mismo vehículo a través de cámaras no contiguas sin solape espacial.
* **Problemas atacados**: Discriminación ineficaz de vehículos con características globales idénticas.
* **Limitaciones de ese entonces**: Los descriptores globales no logran capturar detalles particulares como calcomanías, soportes de techo o daños físicos menores.
* **Soluciones alcanzadas**: Arquitecturas que combinan descriptores locales basados en partes distintivas con atenciones de grafos espaciales e información auxiliar de orientación.

#### 3.3 Tracking
Seguimiento temporal y correspondencia de identidad de objetos en videos.

##### 3.3.1 Single-object tracking (SOT)
Seguimiento de un único objeto de interés específico.
* **Problemas atacados**: Deriva del rastreador (drift) causada por cambios repentinos de iluminación y oclusiones temporales.
* **Limitaciones de ese entonces**: Los filtros de correlación tradicionales acumulan errores y fallan en secuencias largas, mientras que las redes siamesas estándar demandan recursos computacionales excesivos en hardware integrado.
* **Soluciones alcanzadas**: Modelos como SiamDMU que actualizan el template dinámicamente incorporando información semántica y de movimiento de largo plazo.

##### 3.3.2 Multi-object tracking (MOT)
Seguimiento simultáneo de múltiples vehículos y peatones.
* **Problemas atacados**: Pérdida de identidad e ID switches masivos en escenas congestionadas viales.
* **Limitaciones de ese entonces**: Los enfoques desacoplados sufren de sobrecosto computacional secuencial y propagación de errores de detección. Los enfoques conjuntos experimentan conflictos de aprendizaje de características entre la subtarea de detección y la de Re-ID.
* **Soluciones alcanzadas**: Desarrollo de modelos basados en transformadores espacio-temporales que modelan explícitamente las interdependencias mutuas con atenciones de consulta.

#### 3.4 Performance Evaluation

##### 3.4.1 Datasets for low-level perception
* **Problemas atacados**: Falta de un catálogo sistematizado de datos para entrenar tareas viales.
* **Limitaciones de ese entonces**: Dispersión y falta de estandarización en las bases de datos de seguimiento y detección vial.
* **Soluciones alcanzadas**: Compilación detallada de 16 datasets (UA-DETRAC, VeRI-776, Stanford Cars, VisDrone) indicando tamaños, formatos de sensores y propósitos.

##### 3.4.2 Metrics and performance evaluation
* **Problemas atacados**: Inconsistencia y métricas inadecuadas para evaluar la calibración en tareas tridimensionales.
* **Limitaciones de ese entonces**: Evaluar con métricas 2D clásicas (mAP) oculta deficiencias graves en la estimación de la profundidad 3D.
* **Soluciones alcanzadas**: Estandarización de métricas 3D específicas por benchmark (ATE, ASE, ACS, AGS) y métricas de seguimiento MOTA e IDF1.

### 4. High-Level Traffic Perception Tasks

#### 4.1 Traffic Parameter Estimation
Conversión de píxeles visuales en mediciones físicas estructuradas útiles para la ingeniería de tránsito.

##### 4.1.1 Camera Calibration
Estimación de matrices intrínsecas y extrínsecas para mapear coordenadas 2D a 3D.
* **Problemas atacados**: Dependencia de marcas viales y geometrías rígidas artificiales para calibrar cámaras de tráfico en campo.
* **Limitaciones de ese entonces**: Métodos activos o de tablero de ajedrez son inviables de realizar de forma continua en autopistas operativas.
* **Soluciones alcanzadas**: Algoritmos de auto-calibración online basados en puntos de fuga dinámicos de trayectorias y aprendizaje automático de puntos clave vehiculares.

##### 4.1.2 Speed estimation
Medición de la velocidad física instantánea de vehículos en movimiento.
* **Problemas atacados**: Distorsión de perspectiva geométrica que altera el cálculo de distancias recorridas en el plano del sensor.
* **Limitaciones de ese entonces**: Métodos de línea virtual fallan ante cambios de carril y son sensibles al ángulo de cabeceo de la cámara.
* **Soluciones alcanzadas**: Uso de homografías dinámicas calculadas por redes que transforman trayectorias viales al plano del mundo real, reduciendo el margen de error.

##### 4.1.3 Vehicle counting
Conteo volumétrico de flujo vehicular diferenciado.
* **Problemas atacados**: Conteo erróneo en congestión severa debido al solape de vehículos en la proyección bidimensional.
* **Limitaciones de ese entonces**: Los métodos basados en seguimiento duplican o pierden vehículos al reasignar IDs erróneos, y los métodos de regresión de densidad no pueden desglosar flujos direccionales ni por carril.
* **Soluciones alcanzadas**: Clasificación de métodos en basados en seguimiento y de regresión directa, sugiriendo su selección complementaria.

#### 4.2 Traffic Anomaly Detection
Identificación automática de incidentes de tráfico, colisiones y comportamientos viales prohibidos.

##### 4.2.1 Weakly supervised traffic anomaly detection (WSTAD)
Uso de anotaciones débiles que indican la presencia de una anomalía en algún momento del video, sin precisar el instante ni la posición exacta.
* **Problemas atacados**: El costo prohibitivo y la inviabilidad de etiquetar fotograma a fotograma millones de horas de videos de accidentes.
* **Limitaciones de ese entonces**: Los modelos colapsan o se sobreajustan debido a que los datos de accidentes son escasos y las anotaciones viales son ruidosas.
* **Soluciones alcanzadas**: Algoritmos basados en Aprendizaje de Instancias Múltiples (MIL) y redes de atención temporal que localizan temporalmente la anomalía mediante clasificaciones dinámicas por bolsas.

##### 4.2.2 Unsupervised traffic anomaly detection (UTAD)
Modelado puramente de patrones normales para detectar anomalías por desviación sin usar datos previos de accidentes.
* **Problemas atacados**: Incapacidad de predecir o detectar anomalías desconocidas o de las que no se tienen datos previos.
* **Limitaciones de ese entonces**: Los codificadores automáticos simples sufren de "sobre-generalización", logrando reconstruir con bajo error fotogramas que contienen colisiones.
* **Soluciones alcanzadas**: Autoencoders condicionados con módulos de memoria explícita de normalidad (MemAE) y predictores de fotogramas futuros basados en GANs/Transformers.

#### 4.3 Traffic Behavior Understanding
Modelado semántico de las acciones presentes y futuras de los actores de la vía.

##### 4.3.1 Vehicle Behavior Understanding (VBU)
Reconocimiento y predicción de la trayectoria del vehículo y maniobras de giro/cambio de carril.
* **Problemas atacados**: Estimación imprecisa de maniobras de riesgo de colisión en intersecciones ciegas.
* **Limitaciones de ese entonces**: Los modelos puramente cinemáticos no modelan la interacción mutua entre vehículos y colapsan en plazos mayores a 2 segundos.
* **Soluciones alcanzadas**: Redes Transformer espacio-temporales aplicadas a la infraestructura que integran atenciones cruzadas sociales para predecir trayectorias.

##### 4.3.2 Vulnerable Road User Behavior Understanding (VRBU)
Detección de la intención de cruce y trayectoria futura de peatones y ciclistas.
* **Problemas atacados**: Alta tasa de atropellos en pasos de cebra debido a la incapacidad de prever giros repentinos del peatón.
* **Limitaciones de ese entonces**: Evaluar solo la trayectoria histórica no captura la intención real del usuario.
* **Soluciones alcanzadas**: Enfoques híbridos que fusionan poses clave 3D estimadas en tiempo real con la trayectoria del peatón y contexto del entorno vial.

#### 4.4 Performance evaluation

##### 4.4.1 Datasets for high-level perception
* **Problemas atacados**: Escasez de bases de datos estandarizadas para anomalías y comportamiento en vistas de infraestructura.
* **Limitaciones de ese entonces**: La mayoría de los datasets de comportamiento peatonal se recopilan desde cámaras a bordo y no desde postes fijos de vigilancia.
* **Soluciones alcanzadas**: Recopilación de 17 conjuntos de datos clave (CDD, CADP, BrnoCompSpeed, JAAD, PIE, V2X-Seq), documentando sus características.

##### 4.4.2 Metrics and performance evaluation
* **Problemas atacados**: Evaluación dispar del error de predicción espacial a lo largo del tiempo.
* **Limitaciones de ese entonces**: Evaluar con mAP no captura el desvío físico en metros de las predicciones de trayectoria.
* **Soluciones alcanzadas**: Estandarización de métricas ADE (Average Displacement Error) y FDE (Final Displacement Error) para trayectorias, y curvas ROC-AUC para anomalías.

### 5. Limitation Analysis and Future Outlook

#### 5.1 Limitation Overview
Consolidación de las cinco barreras de los sistemas actuales: Degradación perceptiva, Restricciones de datos, Brecha semántica, Cobertura de sensores, y Demandas de cómputo.
* **Problemas atacados**: Vulnerabilidad operativa e inviabilidad de despliegue a gran escala del TSS inteligente.
* **Limitaciones de ese entonces**: Las soluciones se diseñaban aisladas sin entender que el ruido visual degrada consecutivamente la predicción semántica de comportamiento.
* **Soluciones alcanzadas**: Formulación clara de las cinco limitaciones para guiar el desarrollo de frameworks de co-diseño hardware-software.

#### 5.2 Current Solutions and Potential Trends
Revisión de soluciones de mitigación: (1) mejora de la percepción, (2) aprendizaje eficiente, (3) comprensión semántica, (4) sensores cooperativos, y (5) cómputo eficiente.
* **Problemas atacados**: Alto coste computacional y desalineación entre la mejora visual y la precisión analítica posterior.
* **Limitaciones de ese entonces**: Algoritmos de súper resolución que mejoran el aspecto visual pero no la precisión del detector downstream en objetos lejanos.
* **Soluciones alcanzadas**: Orientación de tendencias hacia optimizaciones conjuntas hardware-hardware y alineación de pretext-tasks con la dinámica temporal de tráfico.

#### 5.3 Foundation Model Prospects
El impacto transformador de Large Foundation Models.
* **Problemas atacados**: Falta de adaptabilidad de los detectores supervisados ante categorías novedosas y la brecha del razonamiento de seguridad.
* **Limitaciones de ese entonces**: Los detectores tradicionales colapsan ante elementos ausentes en su set de entrenamiento inicial (vocabulario cerrado).
* **Soluciones alcanzadas**: Uso de open-vocabulary detectors y modelos visuales-lenguaje (VLMs) para consultas visuales interactivas (VQA) y análisis de causa raíz en accidentes. Asimismo, modelos de mundo físico (como Sora) para simulación realista de tráfico.

### 6. Conclusion
Resumen del análisis holístico y los desafíos pendientes.
* **Problemas atacados**: Falta de benchmarks multicategoría unificados y estandarizados para evaluar la degradación.
* **Limitaciones de ese entonces**: Inconsistencia en la comparación directa de rendimientos reportados por diferentes autores debido a entornos propietarios de simulación.
* **Soluciones alcanzadas**: Roadmap estructurado que promueve el desarrollo de conjuntos de datos estandarizados viales y aboga por la convergencia de modelos fundacionales comprimidos eficientemente para el borde.
