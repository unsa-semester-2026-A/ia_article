# Oriented Bounding Boxes for Small and Freely Rotated Objects

- **Key**: Zand2022OBB
- **Year**: 2022
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing (TGRS)

## Resumen
Este artículo presenta un método novedoso de detección de objetos diseñado para manejar objetos libremente rotados y de diversos tamaños en imágenes satelitales y de drones, incluyendo objetos extremadamente pequeños (de hasta 2x2 píxeles). A diferencia de los métodos tradicionales de detección que se basan en la regresión continua de coordenadas y ángulos para estimar cajas delimitadoras orientadas (OBB), el enfoque propuesto replantea el problema como una tarea de clasificación pura y segmentación semántica multiescala. Mediante una arquitectura codificadora-decodificadora (DarkNet-RI) con skips y una estrategia de encendido/apagado (on-off) basada en el tamaño de las celdas, la red predice etiquetas semánticas y puntuaciones de confianza por píxel a través de 5 niveles de escala. En el momento de la inferencia, se aplican operaciones morfológicas, detección de contornos (algoritmo de Suzuki) y el algoritmo de rotating calipers para determinar la caja mínima envolvente orientada. El método también integra un término de regularización invariante a la rotación en el plano ($360^\circ$) que fuerza a la red a compartir características similares para orientaciones diversas. Evaluado en xView y DOTA, el método supera a las aproximaciones del estado del arte sin requerir el diseño heurístico de anclas o cómputos pesados de IoU durante la propuesta de regiones.

## Secciones y Subsecciones

### I. Introducción
Presenta los retos clave de la detección de objetos en imágenes de sensores remotos: variación de escala extrema, alta densidad en escenas complejas y orientaciones aleatorias en 360 grados debido a la perspectiva cenital.
* **Problemas atacados**: Detección ineficiente de objetos muy pequeños y desalineaciones severas entre las cajas delimitadoras horizontales tradicionales (HBB) y los objetos orientados en escenas densas.
* **Limitaciones de ese entonces**: Los detectores tradicionales (ej. Faster R-CNN, YOLO, SSD) usan regresión de cajas axiales y no capturan adecuadamente orientaciones arbitrarias. Las propuestas orientadas basadas en regresión añaden inestabilidad por la ambigüedad en los ángulos.
* **Soluciones alcanzadas**: Propuesta de una arquitectura CNN (DarkNet-RI) que extrae características multiescala y resuelve la detección OBB como un problema de clasificación pura sin depender del diseño manual de anclas o regresores complejos.

### II. Trabajo Relacionado
Analiza la literatura existente dividiéndola en métodos generales de detección y técnicas específicas de sensores remotos.

* **Problemas atacados**: Falta de robustez en detectores generales ante imágenes cenitales con aglomeraciones densas de objetos diminutos.
* **Limitaciones de ese entonces**: Métodos generales (YOLO, Faster R-CNN) imponen restricciones espaciales rígidas de rejilla o anclas que provocan que objetos muy juntos (menores a 32 px en YOLOv3) se ignoren o fundan en una sola caja.
* **Soluciones alcanzadas**: Modelar la detección aérea usando cajas orientadas OBB (ej. RRPN, RoI Transformer, SCRDet) y justificar el uso de aprendizaje de representación multiescala e invariante a la rotación en el plano mediante regularizadores.

#### II.A Métodos de Detección de Objetos Generales
Describe algoritmos basados en regiones (R-CNN, Fast, Faster R-CNN) y de un solo paso (YOLO, SSD), detallando sus mecánicas de propuestas y restricciones espaciales.
* **Problemas atacados**: Dificultades de localización simultánea de clase y caja en imágenes convencionales.
* **Limitaciones de ese entonces**: R-CNN repite cómputos ineficientemente. YOLOv3 reduce imágenes por un factor de 32, impidiendo diferenciar objetos con centroides adyacentes estrechos.
* **Soluciones alcanzadas**: Introducción de capas de alineamiento de región y pirámides de características para mitigar pérdidas de resolución.

#### II.B Detección de Objetos en Sensores Remotos
Tabula y compara 17 metodologías previas que abordan HBB u OBB en imágenes de satélites o drones (Tabla I).
* **Problemas atacados**: Mitigar los sesgos geométricos y de orientación.
* **Limitaciones de ese entonces**: Métodos de OBB satelitales (ej. RoI Transformer, SARD, CAD-Net) dependen casi en su totalidad de la regresión continua de ángulos y coordenadas de esquina, lo cual sufre de ambigüedad en la definición del target (ej. ordenamiento de vértices).
* **Soluciones alcanzadas**: Reemplazar la regresión continua por una clasificación pixel-wise que deduce las coordenadas de la caja orientada mediante geometría computacional en la salida segmentada.

### III. Método Propuesto
Describe el flujo de DarkNet-RI en tres componentes principales: segmentación semántica multiescala, determinación geométrica de cajas orientadas y refinamiento final de cajas.

* **Problemas atacados**: Formulación inestable de regresores de 5 o 8 parámetros en cajas orientadas y control del solapamiento.
* **Limitaciones de ese entonces**: Los regresores continuos sufren ante rotaciones periódicas ($0$ y $180$ grados se ven similares pero causan altos gradientes).
* **Soluciones alcanzadas**: Transformación del problema a predicciones por píxel unificadas mediante morfología matemática y eliminación de falsos positivos en múltiples niveles jerárquicos de escala.

#### III.A Configuración del Problema
Define la formulación de una caja bi-dimensional orientada $b_i = (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4, c_i)$ y la orientación de la caja $R_i$.
* **Problemas atacados**: Representación unificada de objetos rotados en el plano bidimensional.
* **Limitaciones de ese entonces**: Cajas horizontales normales capturan exceso de píxeles de objetos vecinos en distribuciones congestionadas.
* **Soluciones alcanzadas**: Modelado implícito de la rotación resolviendo las esquinas a partir del contorno mínimo que delimita la clase predicha.

#### III.B Segmentación Semántica Multiescala
Detalla el uso de DarkNet-53 con conexiones skip y un decodificador que genera 5 mapas de resolución.
* **Problemas atacados**: Detección unificada de objetos gigantes y extremadamente pequeños en la misma toma.
* **Limitaciones de ese entonces**: El decodificador tradicional suaviza los objetos pequeños perdiendo su rastro en mapas gruesos.
* **Soluciones alcanzadas**: Arquitectura DarkNet-RI que genera predicciones de etiqueta y confianza a 5 niveles de tamaño diferentes.

##### III.B.1 Capa de Aprendizaje de Representación Piramidal
Explica la técnica de encendido/apagado (on-off) basada en el tamaño del objeto con respecto a la celda de la cuadrícula.
* **Problemas atacados**: Interferencia mutua de gradientes entre escalas cuando un objeto diminuto es forzado a aprenderse en resoluciones bajas.
* **Limitaciones de ese entonces**: La pérdida se calcula uniformemente sobre todas las escalas, forzando a capas gruesas a modelar ruido de baja frecuencia de objetos pequeños.
* **Soluciones alcanzadas**: Si un objeto es menor que $1/8$ del tamaño de la celda en una escala $s$, esa escala se apaga ("off") para dicho objeto, delegando su aprendizaje exclusivamente a las capas finas superiores.

##### III.B.2 Invariancia a la Rotación en el Plano
Introduce el regularizador de rotación que mide la similitud L2 de las características de un objeto antes y después de rotarlo.
* **Problemas atacados**: Sesgo del detector ante orientaciones de objetos no presentes en el conjunto de entrenamiento.
* **Limitaciones de ese entonces**: Capas invariantes a rotación explícitas añaden parámetros excesivos y causan sobreajuste (overfitting). El aumento de datos simple no garantiza que la red comparta el mismo mapa de características para diferentes ángulos.
* **Soluciones alcanzadas**: Adición de una pérdida de regularización L2 sobre las áreas solapadas de imágenes rotadas aleatoriamente en un rango de 0 a 360 grados, forzando a la red a mapear vectores idénticos ante variaciones rotacionales sin cambiar la estructura de la red.

##### III.B.3 Entrenamiento
Detalla los términos de la función de coste final: $\ell_s = \ell_{conf} + \ell_{class} + \ell_{rotation}$.
* **Problemas atacados**: Optimización simultánea de confianza de objeto, categoría semántica e invariancia rotacional.
* **Limitaciones de ese entonces**: Costes de regresión de cajas sesgaban el entrenamiento impidiendo la convergencia del clasificador.
* **Soluciones alcanzadas**: Uso de softmax para puntuación de confianza, entropía cruzada multiclase para categorización por celdas, y regularizador L2 para la pérdida de rotación.

#### III.C Determinación de la Caja Orientada
Detalla la conversión morfológica de matrices de etiquetas en cajas orientadas usando el algoritmo de Suzuki y rotating calipers.
* **Problemas atacados**: Extracción geométrica de las OBB a partir de mapas discretos de predicciones semánticas por píxel.
* **Limitaciones de ese entonces**: Las predicciones segmentadas pixel-wise suelen ser ruidosas en los bordes y no proporcionan una caja geométrica directa.
* **Soluciones alcanzadas**: Aplicación secuencial de morfología matemática para denotar regiones conexas, detección de contornos usando el algoritmo de Suzuki, cálculo del convex hull y uso de Rotating Calipers para determinar el rectángulo mínimo envolvente en formato $(x, y, w, h, \alpha)$.

#### III.D Refinamiento de Cajas
Presenta el algoritmo NMS adaptado con umbrales independientes por escala $\{\theta_1, ..., \theta_5\}$ y el uso de puntuación de confianza promediada.
* **Problemas atacados**: Falsos positivos por propuestas duplicadas en bordes de parches o entre escalas.
* **Limitaciones de ese entonces**: El NMS con un solo umbral elimina cajas correctas de tamaño pequeño si se solapan con cajas grandes. Además, NMS estándar usa la confianza máxima de un solo punto en vez de la región.
* **Soluciones alcanzadas**: Ejecución de NMS por capas con umbrales diferenciados. Se descartan cajas de baja calidad promediando la confianza de todas las celdas interiores de la OBB determinada.

### IV. Experimentos
Presenta los resultados empíricos del modelo evaluados en dos conjuntos de datos: xView y DOTA.

* **Problemas atacados**: Validación empírica del rendimiento del modelo frente a detectores basados en regresión.
* **Limitaciones de ese entonces**: Carencia de benchmarks extensos sobre clases con severo desbalance de muestras.
* **Soluciones alcanzadas**: Entrenamiento y testeo en ambos datasets dividiendo las imágenes en parches de 512x512 y comparando el mAP frente al estado del arte.

#### IV.A Conjuntos de Datos y Protocolos
Describe las características de xView (846 imágenes, 60 clases) y DOTA (2,806 imágenes, 15 clases) y los hiperparámetros de entrenamiento.
* **Problemas atacados**: Configuración física y reproducibilidad del experimento.
* **Limitaciones de ese entonces**: Diferencias en los tamaños de los parches y tasas de solapamiento inducían errores de escala en comparaciones cruzadas.
* **Soluciones alcanzadas**: Estandarización de parches a 512x512 con un solapamiento de 10 px. Entrenamiento de 240 épocas usando un lote de 4 en una GPU Titan RTX.

#### IV.B Experimento con el Dataset xView
Evalúa el mAP en las 60 clases globales y un conjunto específico de 19 clases pequeñas e imprecisas.
* **Problemas atacados**: Detección de vehículos e infraestructura pequeña en situaciones de alta densidad urbana y desbalance de clases.
* **Limitaciones de ese entonces**: Los modelos SSD y YOLOv3 no detectan objetos de menos de 10 píxeles por pérdida de resolución espacial.
* **Soluciones alcanzadas**: DarkNet-RI alcanza un mAP récord de 0.3065 en las 19 clases críticas pequeñas y 0.5315 mAP global en 60 clases, superando a YOLT, RFL y SSD. Es capaz de detectar pequeños vehículos de 2x2 píxeles.

#### IV.C Experimento con el Dataset DOTA
Evalúa el desempeño de la tarea OBB en las 15 categorías de DOTA.
* **Problemas atacados**: Validación de la estimación de ángulos y cajas orientadas en DOTA.
* **Limitaciones de ese entonces**: Detectores basados en regresión de 5 parámetros pierden mAP por desalineación de bordes angulares en objetos de gran aspect ratio (ej. barcos, camiones).
* **Soluciones alcanzadas**: DarkNet-RI obtiene un mAP de 75.5%, superando a RoI Transformer (69.56%), SCRDet (72.61%) y SARD (72.95%), liderando en 8 categorías individuales.

#### IV.D Estudio de Ablación
Evalúa el impacto individual de cada componente técnico implementado.

* **Problemas atacados**: Justificar matemáticamente y por rendimiento las decisiones de diseño del DarkNet-RI.
* **Limitaciones de ese entonces**: Redes segmentadoras comunes se aplican directamente sin adaptar a detección.
* **Soluciones alcanzadas**: Pruebas empíricas aislando el regularizador de rotación, la multiescala y el modelo base.

##### IV.D.1 Aprendizaje de Características Invariantes a Rotación
* **Problemas atacados**: Determinar el valor de la pérdida de rotación $\ell_{rotation}$ frente al aumento de datos puro.
* **Limitaciones de ese entonces**: El aumento de datos con rotaciones aleatorias no asegura que el extractor aprenda la misma representación matemática.
* **Soluciones alcanzadas**: Aislar $\ell_{rotation}$ mejora el mAP de 0.5185 a 0.5315, demostrando que el regularizador L2 en el área de solapamiento forza representaciones robustas.

##### IV.D.2 Representación de Características Multiescala
* **Problemas atacados**: Validar la influencia de las conexiones y pérdidas intermedias FC1-FC4.
* **Limitaciones de ese entonces**: Eliminar la pirámide degrada el rendimiento de clases grandes.
* **Soluciones alcanzadas**: Comparación con un baseline de escala única de 256x256 que revela pérdidas severas en la curva ROC y degradación de rendimiento en categorías grandes (Figura 10).

##### IV.D.3 Segmentación Semántica Multiescala
* **Problemas atacados**: Evaluar a DarkNet-RI frente a arquitecturas segmentadoras famosas (UNet, SegNet, SCAttNet).
* **Limitaciones de ese entonces**: Modelos de segmentación estándar no están diseñados para fusionar escalas de manera óptima para cajas delimitadoras.
* **Soluciones alcanzadas**: DarkNet-RI supera ampliamente a UNet (55.4%), SegNet (60.4%) y SCAttNet (61.9%) logrando 75.5% de mAP.

#### IV.E Limitaciones
* **Problemas atacados**: Análisis honesto de fallas cualitativas de DarkNet-RI.
* **Limitaciones de ese entonces**: Los detectores sufren ante objetos muy alargados o sub-objetos anidados.
* **Soluciones alcanzadas**: Identificación de tres fallas principales: 1) Cajas que se parten en dos para objetos muy largos o cortados por el solapamiento del parche, 2) cajas OBB ligeramente desalineadas con los ejes mayores y 3) omisión de objetos pequeños anidados dentro de clases de infraestructura mayor (ej. barcos dentro de un puerto).

### V. Conclusión
* **Problemas atacados**: Resumen de los aportes teóricos y líneas de trabajo futuro.
* **Limitaciones de ese entonces**: La complejidad computacional y sesgos de anclas limitan la generalización a otros dominios.
* **Soluciones alcanzadas**: Demostración de que la detección puede resolverse de forma puramente matemática a partir de predicciones segmentadas discretas, eliminando la necesidad de optimizar anclas y el cálculo de IoU pesado en el entrenamiento. Se sugiere explorar extractores de características jerárquicos en el futuro para clasificar subclases complejas (ej. tipos de camiones).
