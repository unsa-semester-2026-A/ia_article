# Vehicle Detection and Tracking using YOLO and DeepSORT

- **Key**: BinZuraimi2021
- **Year**: 2021
- **Venue**: ISCAIE

## Resumen
Este artículo presenta el diseño y desarrollo de un sistema de visión por computadora para la detección, clasificación y seguimiento de vehículos en tiempo real, orientado a la gestión inteligente del tráfico y la mitigación de la congestión en carreteras de Malasia. El sistema combina el detector de objetos **YOLOv4** (y su versión ligera **YOLOv4-tiny**) con el algoritmo de seguimiento multiobjetivo **DeepSORT** sobre la plataforma TensorFlow. Los autores entrenaron modelos personalizados utilizando un dataset de 7,319 imágenes recopiladas de forma automatizada mediante el toolkit de Open Images v4 (OIDv4), clasificando los vehículos en cuatro clases: automóvil, motocicleta, autobús y camión. En las evaluaciones experimentales utilizando una GPU de gama media (GTX 1660ti), el modelo YOLOv4 personalizado alcanzó una precisión sobresaliente de **82.08% AP50** a una velocidad de procesamiento en tiempo real de **14.12 FPS**. Por otro lado, YOLOv4-tiny demostró ser la mejor opción para hardware de recursos limitados (como Raspberry Pi), alcanzando **76.14% AP50** a una velocidad fluida de **40.11 FPS**. El conteo de vehículos se realiza eficazmente mediante una línea virtual de control integrada en el flujo de tracking.

## Secciones y Subsecciones

### I. Introduction
Introduce el crecimiento del parque automotor en Malasia y la necesidad de automatizar el monitoreo de tráfico mediante Deep Learning para evitar atascos y sus consecuencias negativas.
* **Problemas atacados**: La creciente congestión del tráfico vial en Malasia (31.2 millones de vehículos en 2019) que genera pérdidas de tiempo, contaminación y accidentes. Asimismo, la ineficiencia y fatiga que sufren los operadores humanos al contar y clasificar vehículos manualmente en pantallas de videovigilancia.
* **Limitaciones de ese entonces**: Los métodos de control tradicionales se basan en observaciones manuales subjetivas o bucles magnéticos físicos costosos de instalar y mantener. Los primeros detectores de objetos de visión artificial no lograban equilibrar velocidad y precisión para flujos de video en tiempo real.
* **Soluciones alcanzadas**: Se propone un sistema inteligente de monitoreo que clasifica los vehículos en cuatro categorías y los cuenta al cruzar una línea virtual, usando YOLOv4 para una rápida detección y DeepSORT para un seguimiento estable.

### II. Related Work
Analiza la evolución de los métodos de visión artificial aplicados al tráfico, contrastando técnicas tradicionales con arquitecturas de Deep Learning.
* **Problemas atacados**: La inestabilidad y baja precisión de los primeros clasificadores de vehículos.
* **Limitaciones de ese entonces**:
  * Los métodos tradicionales de visión artificial (como HOG o Haar-like features) no aprenden de forma continua a lo largo del tiempo y registran tasas muy elevadas de falsos positivos.
  * Los detectores de Deep Learning de dos etapas (como Faster R-CNN) generan propuestas de región antes de clasificar, lo que es computacionalmente lento e inviable para inferencia en tiempo real en flujos continuos de autopistas.
* **Soluciones alcanzadas**: Adoptar detectores de una sola etapa (YOLO), los cuales tratan la localización de la caja delimitadora directamente como un problema de regresión de extremo a extremo sobre una cuadrícula de la imagen.

### III. Methodology
Detalla el flujo de trabajo del proyecto, desde la instalación del software hasta la obtención del video de salida procesado.
* **Problemas atacados**: El extenso tiempo requerido para recopilar, etiquetar y entrenar modelos de Deep Learning desde cero en computadores de bajo rendimiento.
* **Limitaciones de ese entonces**: Carencia de flujos de trabajo simplificados para portar modelos entrenados en C (Darknet) hacia entornos de ejecución en Python bajo Windows.
* **Soluciones alcanzadas**: Se propone un flujo estructurado utilizando Git Bash en Windows, descarga automatizada de imágenes de internet, entrenamiento en la nube y conversión a TensorFlow.

#### A. Installation
* **Problemas atacados**: Configuración del entorno de desarrollo Unix en sistemas Windows 10.
* **Limitaciones de ese entonces**: La mayoría de códigos y herramientas de Deep Learning están optimizados para ejecutarse en consolas de Linux o macOS, dificultando su uso nativo en Windows.
* **Soluciones alcanzadas**: Instalación de Git Bash para simular la consola de Linux en Windows y uso de Python como lenguaje central para programar el tracker DeepSORT y la visualización.

#### B. Collection of the images
* **Problemas atacados**: Recopilación rápida de un dataset de imágenes representativo con oclusiones y variedad de ángulos.
* **Limitaciones de ese entonces**: Descargar imágenes de forma manual es ineficiente y no garantiza la variedad necesaria de tipos de vehículos.
* **Soluciones alcanzadas**: Uso de OIDv4 Toolkit para descargar de manera masiva y automatizada imágenes específicas de las clases de interés desde el dataset Google Open Images v4.

#### C. Labelling and classifying
* **Problemas atacados**: Generación de anotaciones de ground truth compatibles con el formato de Darknet/YOLOv4.
* **Limitaciones de ese entonces**: El etiquetado manual consume demasiado tiempo y requiere que los archivos de texto `.txt` sigan un orden estricto de coordenadas normalizadas: `<object-class> <x> <y> <width> <height>`.
* **Soluciones alcanzadas**: OIDv4 Toolkit genera de manera automática los archivos de anotaciones en el formato y estructura requeridos por YOLOv4, acelerando la preparación de datos.

#### D. Training YOLO model
* **Problemas atacados**: Reducción del tiempo de entrenamiento del modelo de detección.
* **Limitaciones de ese entonces**: Entrenar modelos complejos en computadores locales con tarjetas gráficas de consumo (como la GTX 1660ti) toma más de 8 horas para 6,000 iteraciones.
* **Soluciones alcanzadas**: Uso de la plataforma Google Colaboratory (Colab) para entrenar los modelos utilizando GPUs de alto rendimiento (Tesla T4 o P100) en la nube, reduciendo el tiempo de entrenamiento a 4 horas.

#### E. Run code in GitBash
* **Problemas atacados**: Ejecución eficiente del modelo de detección en entornos locales Windows.
* **Limitaciones de ese entonces**: El framework original de Darknet está escrito en C, lo que dificulta su integración directa con librerías de tracking escritas en Python.
* **Soluciones alcanzadas**: Uso de entornos virtuales de Anaconda (Conda) y conversión de los pesos del modelo Darknet (`.weights`) a formato TensorFlow (proceso conocido como DarkFlow) para una inferencia nativa en Python.

#### F. Python program development
* **Problemas atacados**: Asociación temporal de objetos detectados en frames sucesivos y conteo automático.
* **Limitaciones de ese entonces**: YOLO por sí mismo no mantiene la identidad de los vehículos entre frames, por lo que si un vehículo se oculta temporalmente detrás de otro, el sistema pierde su rastro y duplica el conteo.
* **Soluciones alcanzadas**: Integración de **DeepSORT**, que extiende el algoritmo de seguimiento SORT (basado en filtro de Kalman para predicción y algoritmo Húngaro para asociación de IoU) al incorporar un extractor de características de apariencia (deep learning). Esto permite que el sistema recuerde la apariencia de un vehículo y mantenga su ID incluso tras oclusiones temporales. Se añade un script de cruce de línea para contar los vehículos en tránsito.

#### G. Output video
* **Problemas atacados**: Visualización e integración de las estadísticas procesadas.
* **Limitaciones de ese entonces**: Mostrar datos en bruto sin formato visual dificulta la verificación por parte de los operadores.
* **Soluciones alcanzadas**: Generación de un video de salida que renderiza bounding boxes con el nombre de clase y el ID único del vehículo, un contador de FPS y la línea virtual de interés que cambia de color al registrar un cruce.

### IV. Results and Discussion
Muestra las evaluaciones comparativas de los modelos entrenados.
* **Problemas atacados**: Comparación justa del mAP y velocidad de procesamiento de múltiples modelos YOLO.
* **Limitaciones de ese entonces**: Comparar mAP usando diferentes datasets (como el dataset oficial de COCO vs. datasets personalizados pequeños) introduce sesgos debido a la diferencia en la cantidad y resolución de las imágenes.
* **Soluciones alcanzadas**: Se comparan los modelos entrenados bajo el mismo dataset personalizado de vehículos en autopistas de Malasia.

#### A. Datasets
* **Problemas atacados**: Preparación de datos de validación balanceados.
* **Limitaciones de ese entonces**: datasets muy pequeños reducen la capacidad de generalización del modelo.
* **Soluciones alcanzadas**: Construcción de un dataset de entrenamiento de 7,319 imágenes y un conjunto de validación de 750 imágenes por clase (30% de ratio de validación).

#### B. Weight
* **Problemas atacados**: Selección del modelo según limitaciones de almacenamiento.
* **Limitaciones de ese entonces**: Dispositivos embebidos como Raspberry Pi tienen almacenamiento y RAM limitados para cargar modelos de gran tamaño.
* **Soluciones alcanzadas**: Comparación de tamaño de archivos. YOLOv4 e YOLOv3 pesan 250 MB, mientras que sus variantes Tiny pesan apenas 22 MB y 33 MB respectivamente, haciéndolas idóneas para dispositivos embebidos.

#### C. Mean Average Precision
* **Problemas atacados**: Evaluación de la precisión de localización y clasificación.
* **Limitaciones de ese entonces**: Los modelos estándar preentrenados en COCO (80 clases) tienen menor precisión en tareas específicas de tráfico local de autopistas que modelos entrenados con datos locales (4 clases).
* **Soluciones alcanzadas**: YOLOv4 personalizado alcanza 82.08% AP50, superando en 2% a YOLOv3 (80.32%) y por más de 14% a los modelos preentrenados genéricos de AlexAB y Pjreddie en la tarea local de Malasia.

#### D. Performance of model
* **Problemas atacados**: Medición de la velocidad de inferencia en tiempo real en hardware GTX 1660ti.
* **Limitaciones de ese entonces**: Los modelos más precisos suelen ser demasiado lentos para inferencia en tiempo real fluida.
* **Soluciones alcanzadas**: 
  * YOLOv4 alcanza 14.12 FPS (velocidad aceptable para tiempo real y máxima precisión).
  * YOLOv4-tiny alcanza 40.11 FPS (velocidad excelente para hardware ligero a costa de una reducción de precisión de ~6% en mAP).
  * Se comprueba que a mayor complejidad de la red, menor velocidad pero mayor precisión en las predicciones.

### V. Conclusions
Resume las conclusiones y propone recomendaciones para mejorar la arquitectura física del sistema.
* **Problemas atacados**: Despliegue práctico en producción del sistema de monitoreo en carreteras.
* **Limitaciones de ese entonces**: La instalación de computadoras con GPUs pesadas cerca de las cámaras de autopistas es costosa y propensa a daños viales.
* **Soluciones alcanzadas**: Se concluye que YOLOv4 + DeepSORT es una combinación altamente efectiva. Se recomienda desplegar **YOLOv4-tiny** en minicomputadoras **Raspberry Pi** instaladas junto a las cámaras para procesar localmente, o bien transmitir flujos a servidores centrales con GPUs potentes en la nube para procesar con el modelo YOLOv4 completo de mayor precisión.
