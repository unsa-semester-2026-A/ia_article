# Fusion of Deep Sort and Yolov5 for Effective Vehicle Detection and Tracking Scheme in Real-Time Traffic Management Sustainable System

- **Key**: Kumar2023
- **Year**: 2023
- **Venue**: Sustainability

## Resumen
Este artículo propone una solución sostenible y de alto rendimiento en tiempo real para la gestión inteligente del flujo de tráfico en autopistas mediante la integración síncrona del detector You Only Look Once (YOLOv5) y el rastreador Deep Simple Online and Real-time Tracking (Deep SORT). El objetivo del esquema es resolver las limitaciones de precisión debidas a oclusiones, cambios de iluminación y variabilidad de tamaños vehiculares en sistemas de transporte inteligente (ITS). El pipeline se estructura en dos fases: en la primera, YOLOv5 localiza y clasifica los vehículos en cada fotograma generando bounding boxes y extrayendo mapas de características a una tasa de inferencia de 140 FPS; en la segunda fase, Deep SORT procesa estas detecciones extrayendo descriptores de apariencia profunda con una red tipo ResNet. A continuación, asocia los objetos fotograma a fotograma utilizando un filtro de Kalman para predecir las trayectorias espaciales y el algoritmo Húngaro para resolver la asignación binaria combinando solapamiento IoU y distancia de coseno de apariencia. Además, para la estimación del flujo de tráfico, se implementa una lógica de conteo que monitoriza el paso de vehículos a través de "líneas virtuales" y "zonas calientes" (hot zones) configuradas en la escena vial. Evaluado en las bases de datos BDD100K y PASCAL VOC, el modelo integrado alcanza una precisión de 91.25%, un recall de 93.52%, un mAP de 92.18% y una velocidad conjunta de 58 FPS, exhibiendo alta tolerancia frente a oclusiones parciales e inter-vehiculares.

## Secciones y Subsecciones

### 1. Introduction
Presenta el contexto de los sistemas de transporte inteligente sostenibles para mitigar la congestión vial, reducir emisiones de gases y prevenir accidentes en entornos urbanos y de autopista.
* **Problemas atacados**: Lograr un conteo y seguimiento preciso y en tiempo real de múltiples tipos de vehículos que transitan por vías públicas bajo condiciones físicas desafiantes.
* **Limitaciones de ese entonces**: Los métodos clásicos de detección de vehículos basados en aprendizaje automático tradicional sufren de baja adaptabilidad climática. Las soluciones profundas existentes tienen dificultades ante oclusiones temporales, cambios bruscos de iluminación y fluctuaciones visuales (flickering), perdiendo el rastro del vehículo. Adicionalmente, esquemas previos acoplados carecían de suficiente velocidad para su despliegue práctico.
* **Soluciones alcanzadas**: Se propone la fusión de YOLOv5 (por su alta velocidad y pequeño tamaño de pesos) y Deep SORT (por su robustez en la continuidad de trayectorias utilizando descriptores de apariencia profunda), logrando un flujo continuo frame a frame.

#### 1.1. Contribution
* **Problemas atacados**: Proporcionar datos de tráfico precisos, continuos e ininterrumpidos para la toma de decisiones en ITS viales.
* **Limitaciones de ese entonces**: Rastreadores puros como SORT pierden la identidad de los vehículos al ocurrir oclusiones parciales por infraestructura vial o tráfico denso.
* **Soluciones alcanzadas**: Integración de YOLOv5 y Deep SORT. Introduce la asignación de IDs únicos de seguimiento estables que persisten incluso tras oclusiones prolongadas, y define la infraestructura de conteo por zonas calientes.

#### 1.2. Organization
Esbozo estructural del artículo científico en cinco secciones principales.

### 2. Related Work
Estudio de los enfoques teóricos de detección, tracking y arquitecturas de aprendizaje profundo aplicadas al tráfico.
* **Problemas atacados**: Superar el dilema entre velocidad de cómputo y precisión de localización.
* **Limitaciones de ese entonces**: Los detectores de dos etapas (Faster R-CNN, Mask R-CNN) dividen la inferencia en propuesta y clasificación, resultando muy lentos. Los detectores de una sola etapa rápidos (YOLOv3, SSD) a veces sacrifican la precisión con objetos pequeños. En cuanto al tracking, algoritmos clásicos de asociación como MHT y JPDAF requieren altos recursos computacionales. SORT es rápido pero inestable ante oclusiones.
* **Soluciones alcanzadas**: Se demuestra que la fusión de YOLOv5 (que predice cajas mediante regresión directa sobre la imagen) y Deep SORT (que combina métricas de movimiento y de apariencia con CNN) optimiza ambos aspectos.

#### 2.1. Research Study of Vehicle Detection
* **Problemas atacados**: Evolución de las arquitecturas de detección profunda y su idoneidad en sistemas embebidos.
* **Limitaciones de ese entonces**: YOLOv4 posee un archivo de pesos grande (244 MB), dificultando su uso en dispositivos de borde o hardware limitado.
* **Soluciones alcanzadas**: YOLOv5 implementa una estructura de CSPDarknet en el backbone y PANet en el cuello, logrando reducir el archivo de pesos a 27 MB (90% menor que YOLOv4) y alcanzando 140 FPS de velocidad de inferencia, ideal para ITS.

#### 2.2. Research Study of Vehicle Tracking
* **Problemas atacados**: Asociación temporal de objetos superando cambios de perspectiva visual.
* **Limitaciones de ese entonces**: SORT falla cuando ocurren oclusiones o giros bruscos porque asume únicamente coherencia espacial lineal.
* **Soluciones alcanzadas**: Deep SORT soluciona esto extrayendo descriptores visuales profundos (deep appearance metrics) que guardan la "firma" del vehículo, permitiendo reasociarlo correctamente al reaparecer.

#### 2.3. Previous Works
* **Problemas atacados**: Revisión de métodos previos basados en SVM, HOG y filtros de partículas.
* **Limitaciones de ese entonces**: Los clasificadores Haar+Adaboost o HOG+SVM son sensibles a los cambios de escala y la iluminación del entorno real de tráfico.
* **Soluciones alcanzadas**: El uso de redes convolucionales profundas extrae de manera automática características robustas e invariantes a la escala e iluminación.

#### 2.4. Key Consideration
* **Problemas atacados**: Definir los parámetros clave a evaluar: precisión de detección, consistencia de tracking, adaptabilidad de entorno y limitaciones de hardware.

### 3. Proposed Vehicle Detection and Tracking Scheme
Explica la arquitectura unificada y el flujo operacional de los algoritmos de detección y tracking.
* **Problemas atacados**: Diseñar un pipeline de software robusto que combine cinemática física y aprendizaje profundo para la clasificación vehicular.
* **Limitaciones de ese entonces**: El desenfoque por movimiento (blur) de la cámara y las falsas alarmas del detector YOLO.
* **Soluciones alcanzadas**: Se integra YOLOv5 para detección y Deep SORT para tracking, gestionando la asociación mediante Kalman Filter y Hungarian.

#### 3.1. Overview of Proposed Work
* **Problemas atacados**: Flujo lógico de la asociación de datos viales.
* **Limitaciones de ese entonces**: Errores en la predicción lineal de trayectorias.
* **Soluciones alcanzadas**: Se define el flujo frame a frame: YOLOv5 extrae las cajas y puntuaciones, que se transmiten a Deep SORT para el cálculo de similitudes visuales y propagación temporal.

#### 3.2. YOLOv5 Model Overview
* **Problemas atacados**: Diseñar y entrenar la red YOLOv5 en COCO para predecir coordenadas.
* **Limitaciones de ese entonces**: El desvanecimiento de gradiente en redes extremadamente profundas.
* **Soluciones alcanzadas**: Utiliza el bloque CSP Bottleneck para conservar información semántica reduciendo el número de parámetros y FLOPS, y emplea Leaky ReLU y optimizadores SGD/Adam para la convergencia.

#### 3.3. Deep SORT Algorithm for Vehicle Tracking
* **Problemas atacados**: Actualización probabilística y estimación del estado del vehículo en movimiento.
* **Limitaciones de ese entonces**: La incertidumbre en la estimación de velocidad instantánea de vehículos.
* **Soluciones alcanzadas**: Se implementan las ecuaciones del filtro de Kalman (Ecs. 1-5) para modelar y corregir el vector de estado del vehículo $M = [m_x, m_y, a, h]^T$ (posición, relación de aspecto y altura).

#### 3.4. Methodological Flow of Proposed Work
* **Problemas atacados**: Ejecución secuencial de la inferencia viales (Algoritmos 1 a 4).
* **Limitaciones de ese entonces**: Pérdida de trazo (track loss) de vehículos debido a la superposición visual en congestión.
* **Soluciones alcanzadas**: El Algoritmo 1 inicializa y predice las posiciones con Kalman. El Algoritmo 2 utiliza el algoritmo Húngaro para resolver la asignación óptima. El Algoritmo 3 aplica emparejamiento en cascada para evaluar de forma jerárquica la coincidencia temporal e IoU. El Algoritmo 4 unifica todo en un pipeline que lee imágenes y dibuja cajas.

#### 3.5. Vehicle Identification and Vehicle Tracking
* **Problemas atacados**: Corregir y proyectar trayectorias físicas de vehículos en coordenadas espaciales.
* **Limitaciones de ese entonces**: Oclusiones parciales de menos de 0.2 segundos inducen la creación de IDs duplicados.
* **Soluciones alcanzadas**: Se define que si un objeto no tiene solape IoU en un intervalo de 0.2 segundos, se le asigna un nuevo ID, asumiendo que es un vehículo nuevo, y se proyecta la posición de su centroide a coordenadas terrestres (latitud/longitud) para corregir el trazo.

#### 3.6. Vehicle Tracking with Virtual Lines and Hot Zones
* **Problemas atacados**: Contar de manera unívoca los vehículos que se desplazan por carriles específicos.
* **Limitaciones de ese entonces**: Los vehículos que cambian de carril mientras cruzan la línea de conteo son registrados doblemente en los sensores tradicionales.
* **Soluciones alcanzadas**: Se implementa una validación lógica: si un vehículo ya cruzó una zona caliente o línea virtual y cambia de carril, su ID se mantiene en memoria y solo es contabilizado una vez.

#### 3.7. Enhancing the Detection and Tracking of Small Objects Using YOLOv5 and Deep SORT
* **Problemas atacados**: Mejorar la precisión del modelo en la detección de vehículos lejanos o pequeños en la imagen.
* **Limitaciones de ese entonces**: Los detectores tradicionales fallan al procesar objetos de pocos píxeles de resolución.
* **Soluciones alcanzadas**: Se implementan pirámides de características (FPN), aumentación de datos multiescala en el entrenamiento, mecanismos de atención y filtrado NMS de alta resolución para enfocar pequeños parches espaciales.

### 4. Evaluation and Performance Results
Presentación de los resultados de simulación en los conjuntos de datos BDD100K y PASCAL VOC.
* **Problemas atacados**: Medir de manera absoluta el error de localización y tracking.
* **Limitaciones de ese entonces**: La falta de datos reales con anotación de tracking obstruido.
* **Soluciones alcanzadas**: Se evalúan de forma cuantitativa precisión ($P$), recall ($R$), F1-score y mAP@0.5.

#### 4.1. Results Discussion
* **Problemas atacados**: Estudiar la resistencia del algoritmo en escenas con oclusiones severas.
* **Limitaciones de ese entonces**: La interrupción de la línea de visión por postes, puentes u otros vehículos rompe el ID de tracking.
* **Soluciones alcanzadas**: Se evaluó el modelo YOLOv5s+Deep SORT a una velocidad global de 58 FPS en hardware comercial. En escenarios de prueba con oclusiones severas (un carro bloqueado por un poste de luz grande y un auto tapado parcialmente por un bus), la identidad del vehículo (IDs 3 y 4) persistió sin alterarse al reaparecer.

#### 4.2. Vehicle Detection Result Analysis
* **Problemas atacados**: Clasificación de categorías vehiculares específicas (autos, autobuses, camiones).
* **Limitaciones de ese entonces**: Ruido visual que distorsiona las clases en tomas diagonales de autopistas.
* **Soluciones alcanzadas**: El entrenamiento en MS COCO y ajustes locales refinaron las clasificaciones, logrando cajas bien delimitadas en autopistas.

#### 4.3. Vehicle Counting and Tracking Result Analysis
* **Problemas atacados**: Evaluar la precisión de conteo en autopistas utilizando las líneas de control.
* **Limitaciones de ese entonces**: La pérdida de fotogramas disminuye el mAP.
* **Soluciones alcanzadas**: La comparativa con el estado del arte en BDD100K (mAP@0.5 = 51.7%) y PASCAL VOC (mAP@0.5 = 81.2%) demostró la superioridad de la propuesta. El sistema unificado de tracking obtuvo una precisión vial final de 91.25%, un recall de 93.52% y un mAP de 92.18% en video, superando a YOLOv4-3SPP y YOLOv3+DeepSORT.

### 5. Conclusions and Future Work
* **Problemas atacados**: Conclusiones generales del estudio y líneas de investigación futuras.
* **Limitaciones de ese entonces**: Aún existen desafíos en oclusiones extremas y condiciones de iluminación nula o nocturna profunda.
* **Soluciones alcanzadas**: Se valida la factibilidad del modelo YOLOv5 + Deep SORT para la gestión de tráfico sostenible a 58 FPS. Como trabajo futuro, se planea: (1) optimizar de forma independiente los hiperparámetros de ambos subsistemas, (2) explorar algoritmos avanzados de atención espacial, y (3) orientar el desarrollo hacia consideraciones de eficiencia energética y sostenibilidad en ITS.
