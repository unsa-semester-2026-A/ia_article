# Align Deep Features for Oriented Object Detection

- **Key**: Han2022S2ANet
- **Year**: 2022
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing (TGRS)

## Resumen
Este artículo presenta S2A-Net (Single-Shot Alignment Network), un detector de un solo paso específicamente diseñado para la detección de objetos orientados en imágenes aéreas. Los autores identifican que los detectores tradicionales sufren de una desalineación severa entre las cajas delimitadoras de anclaje (anchors) y las características convolucionales alineadas con los ejes (axis-aligned). Para solucionar este problema, S2A-Net introduce dos módulos clave: el Módulo de Alineación de Características (FAM) y el Módulo de Detección Orientada (ODM). El FAM utiliza una Red de Refinamiento de Anclas (ARN) para ajustar un único ancla cuadrada inicial por posición en una propuesta rotada de alta calidad, y luego aplica una Convolución de Alineación (AlignConv) que ajusta dinámicamente los puntos de muestreo convolucionales siguiendo el ancla refinada. El ODM emplea Filtros Rotativos Activos (ARF) para codificar información de orientación y generar características tanto sensibles a la rotación (para regresión de cajas) como invariantes a ella (para clasificación). Además, los autores proponen un esquema para realizar inferencias directas sobre imágenes de gran tamaño, evitando el costoso troceado tradicional y logrando un balance óptimo entre velocidad (FPS) y precisión (mAP). S2A-Net alcanza rendimientos del estado del arte en DOTA (79.42% mAP) y HRSC2016 (90.17% / 95.01% mAP) con alta eficiencia de procesamiento.

## Secciones y Subsecciones

### I. Introducción
Presenta el dominio de la detección de objetos orientados en imágenes aéreas (ODAI) y explica por qué los detectores de dos etapas (R-CNN) dominaban la precisión a costa de velocidad, mientras que los detectores de una etapa tradicionales sacrificaban rendimiento debido al desalineamiento espacial.
* **Problemas atacados**: Desalineación severa entre las características de convolución horizontal (con campo receptivo fijo) y los objetos alargados y orientados aleatoriamente. Inconsistencia entre la confianza de clasificación y la precisión de localización.
* **Limitaciones de ese entonces**: El uso de anclas orientadas densas en detectores de una etapa genera costos computacionales y de memoria excesivos. Alternativas como RoI Transformer evitan esto pero dependen de anclas heurísticas rígidas y operaciones de RoI complejas.
* **Soluciones alcanzadas**: Desarrollo de S2A-Net, una red totalmente convolucional de un solo paso que logra el alineamiento de características de forma adaptativa y reduce el número de anclas a una sola caja cuadrada por celda, refinándola dinámicamente mediante AlignConv.

### II. Trabajos Relacionados
Revisa los detectores de una y dos etapas, enfocándose en cómo han lidiado previamente con las variaciones geométricas y de escala en fotos cenitales.

* **Problemas atacados**: Integración ineficiente de módulos de atención geométrica e inconsistencias en la optimización conjunta de regresores y clasificadores.
* **Limitaciones de ese entonces**: Los detectores tradicionales sufren desbalances de fondo y primer plano extremos. Métodos como R3Det intentan resolver la desalineación remuestreando 5 puntos fijos del ancla, pero no alinean la convolución entera con la orientación de la caja.
* **Soluciones alcanzadas**: Justificación de una red unificada de alineamiento convolucional directo guiado por anclas y separación de descriptores invariantes y sensibles a rotación.

#### II.A Detección de Objetos en Imágenes Aéreas
Analiza enfoques de anclas orientadas heurísticas y métodos que deslizan esquinas sobre cajas horizontales (ej. Gliding Vertex).
* **Problemas atacados**: Sobrecarga computacional de emparejamiento de cajas ground-truth con múltiples anclas rotadas.
* **Limitaciones de ese entonces**: Proponer anclas con múltiples ángulos/escalas ralentiza el cálculo de IoU en el entrenamiento. Deslizar esquinas (Gliding Vertex) no extrae características rotadas de forma nativa.
* **Soluciones alcanzadas**: Uso de una sola ancla cuadrada por celda en FAM que se deforma a OBB dinámicamente.

#### II.B Alineación de Características en Detección de Objetos
Compara operaciones basadas en RoI (RoIPooling, RoIAlign, Deformable RoIPooling) con convoluciones guiadas.
* **Problemas atacados**: Costo computacional de las operaciones a nivel de región (Region-wise operations).
* **Limitaciones de ese entonces**: RoIAlign y afines requieren operaciones de interpolación bilineal y pooling complejas por GPU que forman cuellos de botella en inferencia en tiempo real.
* **Soluciones alcanzadas**: AlignConv logra la misma efectividad de alineación geométrica que RoIAlign pero mediante una única operación totalmente convolucional y ligera.

#### II.C Inconsistencia entre Regresión y Clasificación
Aborda el desajuste donde cajas con alto IoU de localización son suprimidas por tener puntuaciones de clasificación bajas (y viceversa).
* **Problemas atacados**: Supresión errónea en NMS debido a la falta de correlación entre precisión de caja y confianza.
* **Limitaciones de ese entonces**: DoubleHead R-CNN y afines separan cabezas pero no modelan la invariancia/sensibilidad rotacional específica requerida para cada tarea.
* **Soluciones alcanzadas**: Incorporación de filtros rotativos ARF en el ODM para bifurcar el aprendizaje: características sensibles a la rotación para guiar la regresión, y pooling de canal para extraer características invariantes para la clasificación.

### III. Método Propuesto
Describe detalladamente los componentes matemáticos de S2A-Net, incluyendo su línea base RetinaNet, AlignConv, FAM y ODM.

* **Problemas atacados**: Formulación matemática del offset en convoluciones según ángulos OBB y diseño de la pérdida multi-tarea.
* **Limitaciones de ese entonces**: Deformable Convolution tradicional aprende offsets de forma no supervisada o débilmente supervisada, muestreando fuera del objeto en zonas aglomeradas.
* **Soluciones alcanzadas**: Derivación matemática explícita de los offsets en AlignConv basándose directamente en el ángulo, ancho y alto de la caja de ancla refinada.

#### III.A RetinaNet como Línea Base
Adapta RetinaNet cambiando la regresión HBB normal por cajas OBB parametrizadas en $(x, y, w, h, \theta)$.
* **Problemas atacados**: Definición de la orientación angular en el rango $[-\pi/4, 3\pi/4]$.
* **Limitaciones de ese entonces**: RetinaNet clásico ignora el ángulo de orientación.
* **Soluciones alcanzadas**: Reemplazo de la sub-red de regresión horizontal por una cabeza OBB compatible con pérdidas de rotación.

#### III.B Convolución de Alineación
Detalla la formulación matemática de AlignConv donde los puntos de muestreo convolucionales se rotan y escalan según el OBB.
* **Problemas atacados**: Desplazamiento y deformación exacta del kernel $3 \times 3$ convolucional.
* **Limitaciones de ese entonces**: Kernels normales muestrean en una rejilla rectangular fija, capturando fondo irrelevante en objetos diagonales y estrechos.
* **Soluciones alcanzadas**: Para cada celda $p$, el offset $o$ en la posición del kernel $r$ se calcula como $L_p^r - p - r$, donde la posición muestreada $L_p^r$ se define multiplicando el vector del kernel por la matriz de rotación del ancla $R(\theta)^T$ y escalándolo por su ancho y alto $(w, h)$ (Ecuación 3).

#### III.C Módulo de Alineación de Características (FAM)
Muestra el flujo que une la Red de Refinamiento de Anclas (ARN) y la Capa de Convolución de Alineación (ACL).
* **Problemas atacados**: Refinamiento y alineación rápida en un solo paso.
* **Limitaciones de ese entonces**: Generar propuestas de alta calidad solía requerir dos o más etapas completas de la red.
* **Soluciones alcanzadas**: ARN predice offsets OBB preliminares de forma muy ligera sobre anclas horizontales simples. ACL usa esta predicción directamente para alinear los mapas de características que se entregan al ODM.

#### III.D Módulo de Detección Orientada (ODM)
Explica la aplicación de Filtros Rotativos Activos (ARF) y la obtención de descriptores.
* **Problemas atacados**: Modelado de la variabilidad rotacional de las características semánticas.
* **Limitaciones de ese entonces**: Forzar la invarianza en todo el detector degrada la capacidad del regresor de predecir el ángulo exacto.
* **Soluciones alcanzadas**: ARF genera mapas con $N=8$ canales orientados. La clasificación usa invarianza rotacional haciendo un pooling de máximo sobre estos canales (Ecuación 6). La regresión mantiene los mapas de orientación directa para preservar la dirección.

#### III.E Red de Alineación de Un Solo Paso (S2A-Net)
Integra el entrenamiento con la pérdida conjunta de FAM y ODM y el proceso de inferencia.
* **Problemas atacados**: Balanceo de pérdidas en un esquema multi-tarea de una sola etapa.
* **Limitaciones de ese entonces**: Desbalances en las detecciones de FAM y ODM podían provocar gradientes destructivos en las primeras épocas de entrenamiento.
* **Soluciones alcanzadas**: Pérdida unificada (Ecuación 8) que combina Focal Loss (clasificación) y Smooth L1 (regresión) tanto para FAM como para ODM, ponderadas por la cantidad de muestras positivas y un coeficiente de balance $\lambda=1$.

### IV. Experimentos y Análisis
Detalla las pruebas sobre los datasets DOTA y HRSC2016.

* **Problemas atacados**: Validación empírica de S2A-Net frente a detectores de uno y dos pasos en velocidad y mAP.
* **Limitaciones de ese entonces**: Falta de estudios de coste computacional preciso (FLOPs y parámetros).
* **Soluciones alcanzadas**: Demostración de que S2A-Net con ResNet-50 supera a RetinaNet básico en precisión reduciendo los GFLOPs totales.

#### IV.A Conjuntos de Datos
Presenta las características de DOTA (15 clases) y HRSC2016 (dataset de barcos altamente alargados).
* **Problemas atacados**: Adaptación de los datasets a las redes.
* **Limitaciones de ese entonces**: Imágenes de HRSC2016 tienen ratios de aspecto extremos de barcos.
* **Soluciones alcanzadas**: Troceado en parches de 1024x1024 en DOTA y redimensión simple a 800x512 en HRSC2016 manteniendo la relación de aspecto.

#### IV.B Detalles de Inferencia y Entrenamiento
Describe la configuración del optimizador SGD y los entornos físicos.
* **Problemas atacados**: Ajuste óptimo de hiperparámetros.
* **Limitaciones de ese entonces**: La inestabilidad por gradientes en las primeras épocas.
* **Soluciones alcanzadas**: Uso de 4 GPUs V100, entrenamiento durante 12 épocas en DOTA (36 en HRSC) con una tasa de aprendizaje inicial de 0.01 y un periodo de calentamiento (warmup) de 500 iteraciones.

#### IV.C Estudios de Ablación
Evalúa el impacto de la profundidad del detector, AlignConv, ARN y ARF.
* **Problemas atacados**: Determinar la contribución de precisión y velocidad de cada módulo propuesto.
* **Limitaciones de ese entonces**: Falta de comparaciones directas de AlignConv contra Deformable Convolution estándar en OBB.
* **Soluciones alcanzadas**: 1) AlignConv supera a Deformable Conv por 2.4% mAP y a GA-DeformConv por 2.79% mAP en DOTA (Tabla II). 2) Se demuestra que ARF por sí solo no aporta mejoras si no se combina con la alineación espacial proporcionada por ACL (Tabla III). 3) Se confirma que profundidades similares en las cabezas de FAM y ODM optimizan el mAP (Tabla IV).

#### IV.D Detección en Imágenes de Gran Tamaño
Analiza la viabilidad de alimentar la imagen original directamente sin recortarla.
* **Problemas atacados**: Lentitud provocada por el troceado y falsos negativos en bordes de parches.
* **Limitaciones de ese entonces**: Detectores clásicos fallan al procesar imágenes originales por desbordamiento de memoria GPU o ineficiencias de anclas.
* **Soluciones alcanzadas**: Inferencia directa sobre imágenes originales usando precisión media (FP16), reduciendo el tiempo de procesamiento en DOTA a 97 segundos para todo el dataset con pérdidas marginales de mAP (Tabla V). Se demuestra que el troceado introduce problemas en los bordes de los objetos (Figura 7).

### V. Conclusión
* **Problemas atacados**: Resumen y perspectivas.
* **Limitaciones de ese entonces**: Los detectores de una sola etapa solían ser considerados imprecisos en orientaciones críticas.
* **Soluciones alcanzadas**: Demostración de que la alineación de características mediante convoluciones especializadas guiadas por anclas refinadas permite a detectores de un paso superar a las complejas redes de dos etapas en ODAI.
