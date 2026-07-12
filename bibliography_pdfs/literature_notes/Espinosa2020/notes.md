# Detection of Motorcycles in Urban Traffic Using Video Analysis: A Review

- **Key**: Espinosa2020
- **Year**: 2020
- **Venue**: IEEE Transactions on Intelligent Transportation Systems

## Resumen
Este artículo de revisión (survey) presenta un análisis estructurado de los algoritmos de visión computacional y aprendizaje profundo empleados para la detección y el seguimiento de motocicletas a partir de cámaras de vigilancia CCTV en el tráfico urbano. Las motocicletas se clasifican como usuarios vulnerables de la vía (VRU) y representan una gran proporción de los accidentes de tráfico en zonas metropolitanas, especialmente en países en desarrollo. La revisión sigue el flujo de procesamiento de video tradicional: (1) Generación de Hipótesis (HG), que cubre técnicas de detección basadas en apariencia y movimiento; (2) Verificación de Hipótesis (HV), que aborda clasificadores tradicionales discriminativos; (3) Seguimiento de Objetos (Tracking), enfocado en la estimación de trayectorias; y (4) el uso de Aprendizaje Profundo (Deep Learning) tanto para detección (Faster R-CNN, YOLO, SSD) como para seguimiento profundo (MOT). Finalmente, los autores abordan la falta de estandarización en bases de datos y métricas, introducen un nuevo dataset público de referencia llamado "Urban Motorbike Dataset" (UMD) (con 318 tracks y más de 56,000 ROIs anotadas) y evalúan en él modelos de referencia como EspiNet, YOLOv3, Faster R-CNN, MDP y DeepSORT.

## Secciones y Subsecciones

### I. Introduction
Se introduce el panorama de la movilidad urbana global, el aumento del uso de motocicletas en países emergentes y su impacto en la seguridad vial y la salud pública como usuarios vulnerables de la vía (VRUs).
* **Problemas atacados**: La necesidad de desarrollar sistemas de transporte inteligentes (ITS) basados en análisis de video para detectar de manera automatizada a peatones, ciclistas y motociclistas para reducir la tasa de fatalidades por colisión.
* **Limitaciones de ese entonces**: A diferencia de la detección de peatones y ciclistas, que cuenta con una amplia investigación y conjuntos de datos maduros (como el Tsinghua-Daimler Cyclist Benchmark), la investigación específica sobre la detección y seguimiento de motocicletas ha sido históricamente muy limitada.
* **Soluciones alcanzadas**: Se propone estructurar el pipeline común de visión artificial (Generación de Hipótesis, Verificación de Hipótesis y Seguimiento) enfocado en motocicletas, evaluando soluciones clásicas y de aprendizaje profundo.

### II. Hypothesis Generation
Describe la etapa inicial de localización y segmentación de regiones de interés (ROIs) candidatas a contener motocicletas.
* **Problemas atacados**: Extraer candidatos a objetos de la escena vial urbana separándolos del fondo en tiempo real.
* **Limitaciones de ese entonces**: Las variaciones extremas en las orientaciones visuales de las motocicletas, las oclusiones mutuas en tráfico denso y los cambios de iluminación que dificultan la segmentación.
* **Soluciones alcanzadas**: Se dividen los enfoques en métodos basados en apariencia (imágenes estáticas) y basados en movimiento (secuencias temporales).

#### A. Methods Based on Appearance
* **Problemas atacados**: Identificar y extraer regiones candidatas de motocicletas a partir de características estáticas de color, textura o forma en fotogramas individuales.
* **Limitaciones de ese entonces**: La inestabilidad de las características artesanales (handcrafted) ante rotaciones, cambios de escala y transformaciones afines del objeto.
* **Soluciones alcanzadas**: Se revisan descriptores visuales específicos que proveen robustez geométrica y espacial.

##### 1) Explicit Shape Approaches
* **Problemas atacados**: Modelar la geometría explícita del vehículo y del conductor para la generación de ROIs.
* **Limitaciones de ese entonces**: La Transformada Circular de Hough (CHT) genera una gran cantidad de falsos positivos en escenas viales congestionadas y falla ante transformaciones afines o proyectivas.
* **Soluciones alcanzadas**: Uso de CHT para detectar formas circulares de neumáticos y cascos de seguridad vial, aislando zonas del vehículo para clasificaciones posteriores.

##### 2) Texture Features
* **Problemas atacados**: Representación robusta del contorno y textura local de las motocicletas y sus cascos bajo fluctuaciones lumínicas.
* **Limitaciones de ese entonces**: Los extractores como SIFT, Dense SIFT (DSIFT) y SURF no codifican relaciones espaciales entre los puntos clave de interés, lo que limita su capacidad descriptiva.
* **Soluciones alcanzadas**: Uso de técnicas de pirámide espacial de palabras visuales para explotar la localización espacial. Se destaca el descriptor de Histograma de Gradientes Orientados (HOG) como el más robusto para discriminar siluetas de motocicletas y cascos en comparación con LBP y SIFT.

##### 3) Geometric Features & 3-D Models
* **Problemas atacados**: Clasificación de motocicletas mediante la comparación de siluetas con representaciones alámbricas 3D generadas por computadora.
* **Limitaciones de ese entonces**: Los modelos 3D alámbricos requieren una calibración de cámara previa y sufren de un colapso en la tasa de detección en tráfico congestionado debido a la superposición visual de vehículos.
* **Soluciones alcanzadas**: Se implementan descriptores que distinguen entre vistas frontales, traseras y laterales. En vista frontal, se utiliza el grosor de la llanta para discriminar entre bicicletas y motocicletas de forma geométrica.

##### 4) Multiple Features
* **Problemas atacados**: Mejorar la tasa de acierto de detección en escenarios con ruido de fondo extremo integrando múltiples descriptores.
* **Limitaciones de ese entonces**: Integrar múltiples características morfológicas y de textura incrementa drásticamente la dimensionalidad y requiere una tarea compleja de ajuste de parámetros.
* **Soluciones alcanzadas**: Fusión de descriptores SURF, HAAR, HOG y Fourier para clasificar cascos y vehículos en carreteras.

##### 5) Other Descriptors
* **Problemas atacados**: Generación de mapas de densidad y detección de motocicletas en aproximaciones por detrás del vehículo.
* **Limitaciones de ese entonces**: Los filtros de Gabor y descriptores basados en simetría y bordes binarios (como Sobel) están restringidos a ángulos de visión muy específicos (vista trasera), fallando en redes CCTV abiertas.
* **Soluciones alcanzadas**: Se aplican bosques aleatorios para conteo indirecto y coincidencia de plantillas (template matching) asistidos por flujo óptico.

#### B. Methods Based on Motion
* **Problemas atacados**: Segmentación rápida de motocicletas en movimiento aislando el fondo de la escena mediante información espaciotemporal.
* **Limitaciones de ese entonces**: En congestión severa o tráfico semiestático (embotellamientos), los modelos de fondo absorben a los vehículos parados, provocando la pérdida de la detección.
* **Soluciones alcanzadas**: Se implementan restas de fondo dinámicas y modelos adaptativos de actualización temporal.

##### 1) Simple Background Subtraction
* **Problemas atacados**: Extraer de forma simple y rápida siluetas móviles en flujos de video continuos.
* **Limitaciones de ese entonces**: Los blobs resultantes sufren de unificación errónea (varios vehículos se mezclan en una sola masa) bajo tráfico denso y fallan cuando los motociclistas no visten casco.
* **Soluciones alcanzadas**: Se aplican calibraciones con líneas paralelas y crecimiento de regiones para estimar alturas físicas de los blobs. Se diseñan métodos de etiquetado de componentes y reducción de ruido para detectar cabezas sin casco.

##### 2) Gaussian Mixture Models (GMM)
* **Problemas atacados**: Segmentar objetos en movimiento manejando cambios dinámicos de luz, sombras y vibraciones de la cámara.
* **Limitaciones de ese entonces**: Alto costo computacional de actualizar mezclas de Gaussianas en cada píxel y dificultad para separar objetos que se mueven a velocidades similares en paralelo.
* **Soluciones alcanzadas**: Implementación de GMM autoadaptativos y eliminación de sombras mediante análisis cromático en el espacio HSV. Se usa MDGKT para tolerar el ruido del viento en cámaras exteriores.

### III. Hypothesis Verification
Consiste en la clasificación supervisada de las ROIs generadas en categorías específicas de vehículos.

#### A. Discriminative Classifiers
* **Problemas atacados**: Aprender fronteras de decisión óptimas entre motocicletas, ciclistas, peatones y automóviles en espacios de características.
* **Limitaciones de ese entonces**: La mayoría de clasificadores tradicionales se evalúan en bases de datos pequeñas no públicas y sufren la "maldición de la dimensionalidad" en proyecciones kernel de alta dimensión.
* **Soluciones alcanzadas**: Se analizan múltiples arquitecturas supervisadas clásicas.

##### 1) Support Vector Machines (SVM)
* **Problemas atacados**: Clasificación binaria (vehículo vs. fondo) y multiclase optimizando el margen de separación.
* **Limitaciones de ese entonces**: Sensibilidad extrema a las oclusiones físicas y degradación del rendimiento en condiciones climáticas adversas (lluvia, nubes) o de noche.
* **Soluciones alcanzadas**: Se demuestra que la combinación de características HOG con SVM de kernel lineal ofrece el mejor compromiso entre exactitud de clasificación y estabilidad.

##### 2) Decision Trees
* **Problemas atacados**: Definir reglas explícitas de decisión basadas en las dimensiones del blob para la clasificación.
* **Limitaciones de ese entonces**: Las reglas obtenidas por árboles de decisión están altamente sobreajustadas a la resolución, altura y perspectiva de una cámara específica.
* **Soluciones alcanzadas**: Uso de poda de árboles para reducir el sobreajuste en la estimación del tamaño de los blobs.

##### 3) Random Forest
* **Problemas atacados**: Mitigar el sobreajuste de los árboles individuales combinando múltiples estimadores en paralelo.
* **Limitaciones de ese entonces**: Requiere una perspectiva superior (top-down) para evitar el solapamiento visual que distorsiona las decisiones de los expertos locales.
* **Soluciones alcanzadas**: Se implementan clasificadores basados en parches (patch-based random forest) que evalúan configuraciones locales del vehículo.

##### 4) K-Nearest Neighbors (k-NN)
* **Problemas atacados**: Clasificación no paramétrica simple de motocicletas basada en la cercanía de características espaciales.
* **Limitaciones de ese entonces**: Dependencia absoluta de la calidad del blob de movimiento, fallando si hay oclusiones o ruido estático en la escena.
* **Soluciones alcanzadas**: k-NN con $k=5$ para refinar clasificaciones provenientes de cámaras híbridas PTZ y omnidireccionales.

##### 5) Artificial Neural Networks (ANN's)
* **Problemas atacados**: Modelar mapeos no lineales complejos entre imágenes de motocicletas y sus etiquetas correspondientes.
* **Limitaciones de ese entonces**: Los algoritmos de retropropagación clásicos requieren gran cantidad de parámetros de ajuste y suelen quedar atrapados en mínimos locales con muestras pequeñas (p. ej. menos de 150 imágenes de entrenamiento).
* **Soluciones alcanzadas**: Redes multicapa (MLP) aplicadas a la clasificación de cascos utilizando características HOG y Hough en carreteras despejadas.

#### B. Other Approaches
* **Problemas atacados**: Incorporar razonamiento aproximado y comparaciones de plantillas dimensionales sin recurrir a entrenamiento supervisado masivo.
* **Limitaciones de ese entonces**: Las plantillas dimensionales fijas preestablecidas a priori fallan si cambia la resolución de la cámara o la distancia focal.
* **Soluciones alcanzadas**: Clasificadores Neuro-Difusos (ANFIS) basados en lógica Takagi-Sugeno-Kang (TSK) para discriminar autos y motos. Uso de votaciones de trayectoria para etiquetar vehículos tras ser seguidos por varios fotogramas.

### IV. Tracking
Establece la correspondencia temporal de las motocicletas detectadas a lo largo de fotogramas secuenciales.
* **Problemas atacados**: Estimación y suavizado de las trayectorias físicas de los vehículos terrestres en el plano de la imagen.
* **Limitaciones de ese entonces**: El supuesto de proximidad espacial mínima falla en intersecciones urbanas congestionadas donde los caminos de los vehículos se cruzan.
* **Soluciones alcanzadas**: Clasificación de algoritmos en rastreadores de puntos (point trackers), basados en núcleo (kernel) y de silueta.

#### A. Kalman Filter Tracking
* **Problemas atacados**: Predecir posiciones futuras reduciendo el área de búsqueda espacial para acelerar el procesamiento.
* **Limitaciones de ese entonces**: Incapacidad para modelar giros bruscos de motocicletas, y colapso del filtro en presencia de oclusiones prolongadas o cruces de trayectorias.
* **Soluciones alcanzadas**: Se incorporan esquemas de propagación de tracks y votación de etiquetas de clase a lo largo de toda la trayectoria temporal para robustecer el anclaje de identificadores.

#### B. Kanade–Lucas–Tomasi (KLT) Feature Tracker
* **Problemas atacados**: Evitar el colapso del track al rastrear puntos característicos locales (como esquinas de Harris) en lugar del cuerpo completo del vehículo.
* **Limitaciones de ese entonces**: KLT clasifica erróneamente vehículos de dos ruedas eléctricos (e-bikes) de alta velocidad como motocicletas debido a la dependencia de perfiles de velocidad lineal preestablecidos.
* **Soluciones alcanzadas**: Implementación del framework MLMP para análisis de comportamiento de trayectorias en cámaras fijas de CCTV.

#### C. Other Methods
* **Problemas atacados**: Suavizar la asociación de trayectorias en oclusiones severas basándose en la física del movimiento del motociclista.
* **Limitaciones de ese entonces**: Los métodos basados en coincidencia de histogramas de color fallan cuando la iluminación de la escena cambia o cuando las motocicletas tienen texturas similares.
* **Soluciones alcanzadas**: Uso del centro de masa del casco del conductor como punto de anclaje cinemático, tolerando errores de posicionamiento mediante fuerzas de desplazamiento virtuales.

### V. Deep Learning
Revisión del impacto del aprendizaje profundo (DL) en la detección y seguimiento automático de motocicletas.
* **Problemas atacados**: Superar la necesidad de diseñar descriptores manuales (handcrafted) mediante la extracción automática de características jerárquicas invariantes.
* **Limitaciones de ese entonces**: Alta demanda computacional en la fase de convolución que requiere aceleración por GPU y gran cantidad de datos anotados para evitar el sobreajuste.
* **Soluciones alcanzadas**: División de arquitecturas en detectores de dos etapas y de una sola etapa.

#### A. Region Proposal Based Detectors
* **Problemas atacados**: Detección y localización precisa de motocicletas mediante la generación automática de regiones candidatas.
* **Limitaciones de ese entonces**: El tiempo de procesamiento elevado debido al diseño de dos etapas (RPN seguido de clasificación), que limita su uso directo en sistemas integrados en tiempo real.
* **Soluciones alcanzadas**: Se evalúa la evolución de R-CNN, Fast R-CNN y Faster R-CNN para la delimitación espacial y clasificación robusta de vehículos vulnerables.

#### B. Single Stage Detectors (SSDs)
* **Problemas atacados**: Optimizar la velocidad de inferencia tratando la localización de objetos directamente como un problema de regresión.
* **Limitaciones de ese entonces**: Dificultades notables para detectar motocicletas de tamaño extremadamente reducido en imágenes de alta resolución debido a la pérdida de características finas.
* **Soluciones alcanzadas**: Uso de arquitecturas YOLO, SSD y RRC para lograr un rendimiento en tiempo real adecuado para plataformas ADAS. Se analizan redes livianas de 5 capas que dividen la clase negativa para facilitar el entrenamiento.

#### C. Other Approaches
* **Problemas atacados**: Integrar segmentación semántica y emparejar detección de movimiento con redes neuronales convolucionales.
* **Limitaciones de ese entonces**: GMM acoplado con CNNs falla si la segmentación inicial por sustracción de movimiento produce blobs ruidosos o incompletos.
* **Soluciones alcanzadas**: Clasificación híbrida mediante extracción de características con AlexNet y clasificación con SVM lineal. Uso de Faster R-CNN acoplado con GMM para detectar infracciones por falta de casco y leer placas mediante Haar Cascades.

#### D. Deep Visual Tracking
* **Problemas atacados**: Modelar la apariencia y los cambios de pose de los objetos a lo largo de las trayectorias mediante descriptores profundos.
* **Limitaciones de ese entonces**: Las maniobras erráticas y giros rápidos de las motocicletas en espacios estrechos desafían los modelos de movimiento de los filtros de partículas profundos.
* **Soluciones alcanzadas**: Clasificación de rastreadores MOT en modelos basados en características profundas, embeddings de apariencia (como DeepSORT) y aprendizaje de extremo a extremo.

### VI. Datasets and Performance Measures
Establece las bases para la evaluación justa y estandarizada de algoritmos de motocicletas.

#### A. Datasets
* **Problemas atacados**: Proveer datos anotados realistas con variabilidad climática y de tráfico urbano para entrenamiento y validación de modelos.
* **Limitaciones de ese entonces**: La mayoría de conjuntos de datos clásicos (VOC, Caltech256, Penn-Fudan) solo contienen motocicletas en vistas laterales estáticas, carecen de oclusiones realistas de tráfico urbano o corresponden a bases de datos privadas no compartidas.
* **Soluciones alcanzadas**: Se revisan conjuntos como Cityscapes y BLVD. Los autores destacan que la falta de bases de datos específicas para motocicletas en CCTV impide el avance de ITS en el tercer mundo.

#### B. Performance Measures
* **Problemas atacados**: Evaluar de forma matemática y estandarizada la calidad de los modelos de detección, clasificación y tracking.
* **Limitaciones de ese entonces**: El uso inconsistente de métricas entre diferentes publicaciones científicas que imposibilita la comparación directa de rendimiento.
* **Soluciones alcanzadas**: Se definen formalmente las métricas clave.

##### 1) Detection Measures
* **Problemas atacados**: Medir el error de localización y conteo de cuadros delimitadores.
* **Limitaciones de ese entonces**: Las métricas FPPW pueden dar una falsa sensación de precisión al no ponderar el área espacial del objeto.
* **Soluciones alcanzadas**: Estandarización sobre la base del mAP (mean Average Precision) derivado del cálculo del área de intersección sobre unión (IoU) y la métrica GAME para conteo sobre celdas de rejilla.

##### 2) Classification Measures
* **Problemas atacados**: Evaluar el rendimiento en conjuntos de clases altamente desbalanceados (p. ej., pocas motocicletas frente a muchos automóviles).
* **Limitaciones de ese entonces**: La precisión simple de clasificación es engañosa en conjuntos de datos viales donde la clase mayoritaria enmascara fallas.
* **Soluciones alcanzadas**: Adopción formal de la métrica de F1-Score acoplando precisión y recall para evaluar matrices de confusión multiclase.

##### 3) Tracking Measures
* **Problemas atacados**: Cuantificar las fallas de asociación temporal de objetos múltiples.
* **Limitaciones de ese entonces**: Las métricas tradicionales de tracking no capturan fallas inducidas por una mala inicialización en el primer frame.
* **Soluciones alcanzadas**: Uso del protocolo CLEAR MOT que define formalmente MOTP (precisión posicional promedio de las cajas) y MOTA (tasa de acierto de seguimiento considerando fallas de omisión, falsas alarmas e ID switches), complementado con la métrica PR-MOTA.

### VII. A Baseline for Future Researchers
Presenta la contribución de datos y resultados base de la investigación de los autores.

#### A. The Urban Motorbike Dataset
* **Problemas atacados**: Purgar la escasez de datos abiertos anotados de motocicletas en entornos urbanos caóticos tomados desde perspectivas aéreas.
* **Limitaciones de ese entonces**: La ausencia de conjuntos de datos de video que incluyan anotaciones espaciales en condiciones reales de viento e inclinación de cámara.
* **Soluciones alcanzadas**: Se introduce el dataset de acceso público "Urban Motorbike Dataset" (UMD) que incluye 318 trayectorias y 56,975 ROIs anotadas tomadas desde un dron Phantom 4, donde el 60% de los datos incluye oclusiones complejas de motocicletas en tráfico denso.

#### B. Preliminary Evaluation
* **Problemas atacados**: Establecer benchmarks y líneas base iniciales de detección y seguimiento sobre el nuevo dataset UMD.
* **Limitaciones de ese entonces**: Evaluar modelos de forma aislada sin acoplarlos en arquitecturas TBD completas.
* **Soluciones alcanzadas**: Se comparan detectores y rastreadores populares sobre UMD.

##### 1) Detection
* **Problemas atacados**: Comparar la exactitud de localización de modelos de aprendizaje profundo entrenados desde cero en UMD.
* **Limitaciones de ese entonces**: Faster R-CNN y YOLOv3 fallan al localizar motocicletas muy pequeñas u ocluidas.
* **Soluciones alcanzadas**: Se evalúa "EspiNet" (modelo propio), logrando un AP de 88.8% y F1-score de 91.8%, superando notablemente a YOLOv3 (64.9% AP) y Faster R-CNN (61.9% AP) en la base UMD.

##### 2) Tracking
* **Problemas atacados**: Comparar el rendimiento de asociación temporal utilizando diferentes detectores de entrada sobre el dataset UMD.
* **Limitaciones de ese entonces**: El tracker DeepSORT y los modelos basados en MDP dependen críticamente de la precisión del detector frontal.
* **Soluciones alcanzadas**: Se demuestra que EspiNet como detector mejora el desempeño de DeepSORT (alcanzando 72.3% MOTA) y MDP (73.7% MOTA) en UMD en comparación con el uso de YOLO o Faster R-CNN, demostrando la importancia de la calidad del detector inicial.

### VIII. Discussion
* **Problemas atacados**: Analizar los cuellos de botella tecnológicos restantes en la detección y seguimiento de motocicletas.
* **Limitaciones de ese entonces**: La ineficiencia de las ventanas deslizantes (sliding windows) clásicas y la dificultad para adaptar detectores de sistemas a bordo (ADAS) a la perspectiva picada de cámaras fijas CCTV urbanas.
* **Soluciones alcanzadas**: Se recalca que el uso de calibración de cámara para preestablecer ROIs reduce significativamente el espacio de búsqueda y acelera la inferencia. Se discuten las limitaciones físicas causadas por oclusiones extremas nocturnas e inclemencias climáticas.

### IX. Conclusions and Future Work
* **Problemas atacados**: Consolidar las directrices de investigación para mejorar la protección de los usuarios de motocicletas.
* **Limitaciones de ese entonces**: La literatura actual se concentra mayormente en entornos limpios o autopistas despejadas con modelos completamente supervisados que dependen de costosas anotaciones manuales.
* **Soluciones alcanzadas**: Se concluye que el aprendizaje profundo es la vía óptima para ITS, pero requiere el desarrollo de técnicas de entrenamiento semisupervisado o no supervisado. Se propone avanzar en la detección y tracking incorporando razonamiento contextual de la escena urbana y expandiendo los datasets a condiciones nocturnas y de baja visibilidad.
