# A Scalable Real-Time Multi-Camera Vehicle Tracking System for Urban Environments

- **Key**: Alas2026
- **Year**: 2026
- **Venue**: IEEE Access

## Resumen
Este artículo presenta un sistema escalable y de baja latencia para el seguimiento de vehículos con múltiples cámaras (MCVT) en entornos urbanos y centros de distribución logística. A diferencia de los enfoques centralizados monolíticos tradicionales que saturan el ancho de banda y crean cuellos de botella, la arquitectura propuesta se despliega de manera distribuida sobre el continuo de computación de borde y niebla (edge-fog continuum). El flujo de procesamiento se divide en tres fases principales: (1) detección y seguimiento unicámara liviano en el borde mediante visión computacional tradicional sin GPU, empleando un enlazador de centroides basado en dirección (direction-aware centroid linker) para formar tracklets estables; (2) extracción de descriptores visuales en la niebla donde un modelo YOLO refina las cajas eliminando falsos positivos y clasificando el tipo de vehículo (carro, van, camión, montacargas, motocicleta) y un Vision Transformer (ViT-B/16) preentrenado y ajustado con triplet loss y LoRA genera embeddings de apariencia compactos; y (3) reconstrucción de trayectorias intercámara en un nodo central aplicando restricciones de topología vial y viabilidad del tiempo de viaje. Validado en el complejo Mercabarna con 27 cámaras CCTV y 7 nodos de borde, el sistema procesa video a 20-29 FPS por flujo bajo limitaciones estrictas de cómputo y logra identificar correctamente más del 80% de los emparejamientos de vehículos.

## Secciones y Subsecciones

### I. Introduction
La introducción justifica la relevancia de los sistemas de seguimiento de vehículos multicámara (MCVT) en tiempo real para la gestión del tráfico urbano y grandes centros de distribución.
* **Problemas atacados**: Diseñar y desplegar sistemas de seguimiento continuo de vehículos a través de redes heterogéneas de CCTV en tiempo real. Esto involucra manejar cambios de punto de vista, resoluciones de imagen variables, ruido ambiental, oclusiones frecuentes en tráfico denso y la difícil tarea de asociar identidades entre cámaras disjuntas sin solapamiento de vista.
* **Limitaciones de ese entonces**: Los enfoques monolíticos o basados en la nube sufren de cuellos de botella en el ancho de banda al transmitir múltiples transmisiones de video bruto y representan un único punto de falla. Por otro lado, las arquitecturas basadas puramente en aprendizaje profundo para el seguimiento (como rastreadores basados en Transformers como MOTR o TransTrack) son demasiado pesadas para operar en dispositivos de borde con recursos computacionales y de memoria limitados (CPU/GPU restringidos).
* **Soluciones alcanzadas**: Se propone una tubería (pipeline) de MCVT modular, eficiente y de baja latencia que distribuye la computación entre el borde (edge) y la niebla (fog). Se divide el proceso en: detección ligera en el borde, extracción de características semánticas y asociación multicámara con restricciones de espacio y tiempo físico, logrando un sistema escalable con degradación elegante.

### II. Related Work
Sintetiza la literatura en tres áreas fundamentales: seguimiento de objetos múltiples (MOT) en una sola cámara, seguimiento multicámara (MCT) y modelos de reidentificación de vehículos (ReID).

#### A. Multi-Object Tracking
* **Problemas atacados**: Asociación temporal de detecciones de objetos dentro de un único flujo de video para conformar trayectorias coherentes.
* **Limitaciones de ese entonces**: Los rastreadores clásicos como SORT son veloces pero vulnerables a oclusiones y cambios abruptos de velocidad. Las mejoras como DeepSORT y modelos basados en Transformers (TrackFormer, MOTR, MOTRv2) logran gran precisión pero requieren aceleración GPU masiva, sobrecargando el hardware de borde comercial.
* **Soluciones alcanzadas**: Se destacan estrategias de asociación livianas sin backbones profundos para el seguimiento de movimiento, como ByteTrack (política de doble umbral para detecciones de baja confianza) y OC-SORT, que mejoran la robustez frente a oclusiones y movimientos no lineales de manera computacionalmente eficiente.

#### B. Multi-Camera Tracking
* **Problemas atacados**: El enlace de trayectorias individuales calculadas por cámaras separadas geográficamente (inter-camera association).
* **Limitaciones de ese entonces**: La mayoría de las soluciones de gran escala requieren extracciones visuales muy complejas. Las aproximaciones basadas en Bird's Eye View (BEV) o V2X requieren calibración extrínseca precisa de cámaras superpuestas, lo cual no es factible en redes de CCTV urbanas antiguas compuestas por cámaras montadas asimétricamente y sin solapamiento. Las técnicas basadas puramente en reconocimiento de matrículas (LPR) fallan cuando estas son ilegibles o están ocultas en movimiento.
* **Soluciones alcanzadas**: Se revisan enfoques de agrupamiento de grafos basados en afinidades de apariencia y restricciones espaciotemporales (como CityFlow). Se discuten soluciones distribuidas que comunican únicamente metadatos de baja dimensión a través de redes inalámbricas de baja potencia para preservar la privacidad y reducir el ancho de banda.

#### C. ReIdentification Models
* **Problemas atacados**: Obtención de representaciones de apariencia (embeddings) que discriminen vehículos individuales a pesar de grandes cambios de iluminación, pose y ángulo.
* **Limitaciones de ese entonces**: Los modelos estándar entrenados en conjuntos genéricos sufren deDomain Shift cuando se despliegan en entornos reales específicos. Además, las imágenes de cámaras urbanas presentan desenfoque por movimiento y baja resolución, lo que degrada la estabilidad del emparejamiento por apariencia.
* **Soluciones alcanzadas**: Se resalta el uso de Vision Transformers (ViT) como TransReID con entrenamiento progresivo en múltiples etapas (preentrenamiento supervisado/autosupervisado, ajuste en datos sintéticos intermedios y sintonización final en datos reales con triplet-loss). También se analizan técnicas de superresolución de rama única o múltiple (MBFSR) como preprocesamiento antes de la extracción de descriptores visuales.

### III. Our Approach
Describe la arquitectura del sistema propuesto y el flujo de comunicación y procesamiento distribuido entre el borde y la niebla.
* **Problemas atacados**: Diseñar un ecosistema de seguimiento vehicular que minimice el uso de hardware de servidor central y distribuya tareas de manera eficiente en la infraestructura edge-fog.
* **Limitaciones de ese entonces**: La ineficiencia de procesar video de alta resolución de forma centralizada y la falta de acoplamiento entre la detección de movimiento y las reglas físicas de tránsito.
* **Soluciones alcanzadas**: Se diseña un sistema desacoplado que envía cultivos de imágenes (crops) de vehículos detectados en el borde hacia la niebla a través de un broker AMQP. Esto disminuye drásticamente el consumo de ancho de banda. La niebla maneja asincrónicamente la validación y clasificación de vehículos con YOLO, la extracción de embeddings con ViT y la reconstrucción global en tiempo real.

#### B. Vehicle Detection Pipeline
* **Problemas atacados**: Detección de candidatos a vehículos en el borde minimizando el costo computacional (ejecutándose únicamente en CPU).
* **Limitaciones de ese entonces**: Los detectores de aprendizaje profundo convencionales consumen demasiada memoria RAM y capacidad de CPU cuando procesan flujos de video completos continuamente.
* **Soluciones alcanzadas**: Se crea un pipeline de cuatro etapas: (1) identificación de cajas mediante sustracción de fondo por modelado de mezclas de Gaussianas y filtrado morfológico; (2) postprocesamiento que agrupa fragmentos mediante búsqueda en profundidad (DFS) en un grafo de componentes conectados; (3) seguimiento en tiempo real; y (4) estimación de dirección.

##### 1) Bounding Box Identification
* **Problemas atacados**: Delinear regiones de movimiento candidatas a vehículos rápidamente sobre la transmisión de video.
* **Limitaciones de ese entonces**: El ruido de imagen, cambios rápidos de iluminación y sombras móviles provocan detecciones erróneas y fragmentación de máscaras.
* **Soluciones alcanzadas**: Se aplica sustracción de fondo basada en píxeles. La máscara binaria obtenida se suaviza tres veces con un kernel de 3x3, se dilata iterativamente para unificar regiones y se filtra con un umbral de intensidad superior a 100 para descartar ruido estático.

##### 2) Post Processing
* **Problemas atacados**: Agrupar múltiples cajas delimitadoras fragmentadas correspondientes a un único vehículo físico.
* **Limitaciones de ese entonces**: Los vehículos grandes o uniformes (como camiones con remolques blancos) solo producen cambios de movimiento en sus bordes, lo que causa múltiples detecciones separadas de una sola entidad, duplicando los tracks posteriores.
* **Soluciones alcanzadas**: Se representa a los rectángulos detectados como nodos en un grafo. Se traza una arista si su intersección (IoU) excede un porcentaje. Se aplica DFS para extraer las componentes conectadas del grafo y unificarlas en una única caja delimitadora candidata por vehículo.

##### 3) Real-Time Tracking
* **Problemas atacados**: Asociar de forma continua las cajas detectadas frame a frame bajo restricciones de tiempo real en CPU.
* **Limitaciones de ese entonces**: SORT no maneja bien oclusiones de largo plazo ni giros en intersecciones con solapamiento espacial.
* **Soluciones alcanzadas**: Se introduce un enlazador de centroides basado en dirección. Se aplica un umbral mínimo de IoU para la asociación inicial. Luego, se calculan vectores de dirección normalizados a corto, mediano y largo plazo utilizando promedios de posiciones pasadas para evitar el jitter. Se evalúa la compatibilidad angular (desviación trigonométrica) antes y después de simular la adición de la nueva caja a la trayectoria. El crop se asigna a la trayectoria que minimiza la desviación angular usando una métrica wrap180, inicializando un nuevo track si no se cumplen las condiciones.

##### 4) Direction Identification
* **Problemas atacados**: Identificar las direcciones de inicio y fin (UP, LEFT, RIGHT, DOWN) de las trayectorias vehiculares de forma consistente a pesar de la orientación y deformación de cada cámara CCTV.
* **Limitaciones de ese entonces**: Cada cámara tiene una inclinación y rotación particular con respecto a la red vial global, por lo que los ángulos absolutos no coinciden entre vistas heterogéneas.
* **Soluciones alcanzadas**: Se implementa un método de etiquetado de dirección mediante interpolación ponderada. Se definen puntos de referencia específicos por cámara con intervalos angulares predeterminados. Para la posición final del vehículo, se calculan pesos normalizados inversamente proporcionales a la distancia euclidiana hacia los puntos de referencia, ponderando los límites del intervalo angular global para etiquetar la trayectoria de movimiento con el eje geográfico correcto.

#### C. Feature Extraction Component
* **Problemas atacados**: Extracción de firmas visuales de alta calidad a partir de los cultivos (crops) filtrados para la posterior reidentificación intercámara.
* **Limitaciones de ese entonces**: Los cultivos iniciales del borde contienen falsos positivos (sombras, peatones, vehículos estacionados) y a veces capturan múltiples vehículos solapados, lo que contamina los descriptores.
* **Soluciones alcanzadas**: Se procesa cada cultivo con un detector YOLO refinado que filtra falsos positivos y clasifica el vehículo en 5 clases. Si detecta múltiples vehículos en un cultivo, los segmenta de forma independiente. Cada cultivo validado se introduce en un transformador visual (ViT-B/16) preentrenado en ImageNet y ajustado mediante LoRA y triplet-loss con el dataset de Mercabarna. Se almacena el embedding promedio obtenido de toda la trayectoria del vehículo.

#### D. Trajectory Reconstruction
* **Problemas atacados**: Unificar las trayectorias unicámara (eventos de cámara) en trayectorias completas del mundo real a través de cámaras disjuntas (eventos de calle).
* **Limitaciones de ese entonces**: La similitud visual intrínseca entre vehículos de flotas idénticas (como camiones logísticos repetitivos o montacargas amarillos) induce a un colapso de precisión en búsquedas globales basadas puramente en la similitud de apariencia de los embeddings.
* **Soluciones alcanzadas**: Se codifican restricciones espaciotemporales a priori basadas en la topología de la red vial y los tiempos de viaje mínimos y máximos físicamente plausibles. La búsqueda de correspondencias de apariencia se restringe a cámaras lógicamente adyacentes dentro de ventanas temporales coherentes. Se adoptan dos umbrales adaptativos de similitud de coseno: un umbral permisivo (0.65) si las etiquetas de dirección del vehículo coinciden con el flujo lógico, y un umbral más estricto (0.85) cuando las direcciones son desconocidas o reversas, mitigando falsas fusiones.

### IV. Experiments and Results
Detalla los resultados empíricos obtenidos en las pruebas de laboratorio y el despliegue físico del sistema.

#### A. YOLO Detection Model
* **Problemas atacados**: Validar la precisión de la detección y clasificación del detector de refinamiento YOLO en el entorno logístico real de Mercabarna.
* **Limitaciones de ese entonces**: La falta de datos locales etiquetados de vehículos inusuales en autopistas como montacargas (forklifts) y camiones logísticos pesados operando en zonas de carga.
* **Soluciones alcanzadas**: Se etiqueta un conjunto de validación de 8,964 imágenes (con 22,474 instancias de vehículos). El modelo YOLO refinado logró un mAP@0.5 sobresaliente para montacargas (debido a su forma distintiva) y camiones, mientras que el menor desempeño (menor recall) se dio en motocicletas debido a oclusiones y tamaño reducido.

#### B. Vehicle Embeddings
* **Problemas atacados**: Demostrar la robustez y capacidad de generalización del espacio de embeddings visuales del ViT.
* **Limitaciones de ese entonces**: Los modelos de ReID tienden a sobreajustar sus representaciones a los sesgos de iluminación y textura de la base de datos de entrenamiento.
* **Soluciones alcanzadas**: Se comparó la similitud de coseno para pares idénticos de vehículos con tres configuraciones de ViT (Base, Stanford, Mercabarna). El modelo Mercabarna afinado logró la mayor densidad de similitud (>0.85) y separación interclase. Para evaluar la generalización en una configuración zero-shot (sin entrenamiento local previo), se testearon los modelos en los benchmarks VeRi-776 y VRIC, superando a los baselines tradicionales de la literatura y validando la consistencia semántica de la representación.

#### C. Reidentification Evaluation Metrics
* **Problemas atacados**: Medir el desempeño del pipeline multicámara completo en condiciones de operación abierta.
* **Limitaciones de ese entonces**: El movimiento altamente dinámico, no lineal, paradas temporales, giros bruscos y reversas de los vehículos degradan las señales de dirección y fragmentan los tracks.
* **Soluciones alcanzadas**: Evaluado sobre 226 emparejamientos etiquetados a mano, el sistema alcanzó un alto recall, aunque mostró una precisión e IDF1 moderados debido a fragmentación de trayectorias (identidades duplicadas) inducida por oclusiones prolongadas o fallos temporales en la detección en zonas ciegas de la red de cámaras.

#### D. Ablation Study
* **Problemas atacados**: Cuantificar el aporte individual de cada componente tecnológico (seguimiento unicámara, embedding, refinamiento YOLO, y restricciones físicas) al éxito del sistema.
* **Limitaciones de ese entonces**: La dificultad de determinar cuál componente optimizar con prioridad para obtener el mayor beneficio en precisión intercámara o continuidad.
* **Soluciones alcanzadas**: El estudio de ablación reveló que: (1) el modelo de embedding es el factor dominante de la calidad de asociación intercámara (IDP/IDR); (2) el seguimiento unicámara influye directamente en la métrica de Unicidad (evitando la fragmentación de tracklets); (3) el refinamiento YOLO actúa como filtro de ruido mejorando la precisión a costa de un ligero recall; y (4) las restricciones de topología y tiempo de viaje son esenciales para evitar el colapso de la precisión visual global en flotas de apariencia homogénea.

#### E. Deployment Test
* **Problemas atacados**: Garantizar el funcionamiento continuo y en tiempo real del sistema MCVT en una infraestructura de producción real.
* **Limitaciones de ese entonces**: Limitación de hardware en el borde (procesamiento en Mini PCs comerciales sin GPUs dedicadas y consumo elevado de memoria RAM en flujos con múltiples vehículos concurrentes).
* **Soluciones alcanzadas**: Se desplegaron 27 cámaras y 7 edge Mini PCs (Intel i7, 32GB RAM). Se logró procesar los flujos a 20-29 FPS en tiempo real. En situaciones de alta carga, se implementó una estrategia de submuestreo aleatorio para mantener el consumo de RAM en el borde en un límite de 3.65 - 5.65 GB por stream. El procesamiento de embeddings y refinamiento en un servidor Xeon central toma entre 1 y 5 segundos por evento vehicular utilizando trabajadores en paralelo.

### V. Conclusion
* **Problemas atacados**: Consolidar las lecciones aprendidas del despliegue práctico del sistema edge-fog.
* **Limitaciones de ese entonces**: El sistema está altamente ajustado (retuned) a la topología y dinámicas de Mercabarna, requiriendo calibraciones no triviales si se desea portar a una ciudad diferente. También depende de luz diurna clara.
* **Soluciones alcanzadas**: Se demuestra que la modularidad permite actualizar de forma independiente cualquiera de los niveles (ej. migrar a detectores de aprendizaje profundo cuantizados para CPU en el borde o incorporar modelos probabilísticos de tiempo de viaje) sin alterar el resto del sistema, ofreciendo una base robusta para ITS escalables de próxima generación.
