# Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges

- **Key**: Ding2022DOTA
- **Year**: 2022
- **Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

## Resumen
Este artículo presenta DOTA (Dataset of Object deTection in Aerial images), un conjunto de datos a gran escala y un benchmark exhaustivo para la detección de objetos en imágenes aéreas (ODAI). La detección de objetos en este dominio es especialmente compleja debido a las grandes variaciones de escala y a las orientaciones arbitrarias de los objetos observados desde una vista cenital. Para solucionar esto, DOTA introduce anotaciones con cajas delimitadoras orientadas (OBB), permitiendo una delimitación precisa de los objetos. En su versión v2.0, DOTA se expande a 11,268 imágenes aéreas y 1,793,658 instancias de objetos distribuidas en 18 categorías comunes. Utilizando este dataset, los autores evalúan 10 algoritmos de detección del estado del arte bajo más de 70 configuraciones unificadas, analizando a fondo aspectos críticos como el diseño de módulos geométricos (RoI Transformer vs. Deformable RoI Pooling), la cantidad de propuestas de regiones y las técnicas de aumento de datos por escala y rotación. También se libera una biblioteca de código y un kit de desarrollo para facilitar la investigación reproducible en el área.

## Secciones y Subsecciones

### 1. Introducción
Presenta el contexto de la detección de objetos en imágenes aéreas (ODAI) y su importancia en aplicaciones reales como rescates de emergencia, agricultura de precisión y gestión urbana. Identifica las principales dificultades del dominio: orientación arbitraria de los objetos, variación de escala masiva, densidad de distribución no uniforme y relaciones de aspecto extremas.
* **Problemas atacados**: La falta de conjuntos de datos a gran escala con anotaciones orientadas que reflejen las condiciones del mundo real, y la carencia de una biblioteca unificada que permita evaluar de forma justa y reproducible los modelos.
* **Limitaciones de ese entonces**: Los detectores convencionales entrenados en imágenes naturales usan cajas delimitadoras horizontales (HBB), inservibles para discriminar objetos densos y orientados (como barcos o vehículos aparcados). Además, los datasets aéreos previos tenían pocas instancias y categorías, y las bibliotecas populares (como MMDetection o Detectron) no admitían detección orientada.
* **Soluciones alcanzadas**: Propuesta del dataset DOTA expandido (v2.0) con anotaciones OBB detalladas, desarrollo de una biblioteca específica y un servidor de evaluación en línea, y construcción de benchmarks integrales con 10 algoritmos evaluados bajo las mismas condiciones de hardware y software.

### 2. Trabajo Relacionado
Revisa los antecedentes en tres subtemas principales: bases de datos para detección horizontal convencional, bases de datos aéreas previas, detectores profundos adaptados al dominio aéreo y bibliotecas de código de detección.

* **Problemas atacados**: Comparación de las características estructurales de DOTA frente a otros datasets, y justificación de la necesidad de modelar la rotación y escala.
* **Limitaciones de ese entonces**: Los datasets de imágenes naturales (PASCAL VOC, MS COCO, ImageNet) no tienen vistas cenitales ni anotaciones de rotación. Los datasets de imágenes aéreas anteriores (NWPU VHR-10, VEDAI, etc.) contaban con muy pocas instancias u omitían el uso de OBB, limitándose a HBB.
* **Soluciones alcanzadas**: Posicionamiento de DOTA como el mayor dataset del dominio con anotaciones OBB, alta densidad de instancias por imagen (promedio de 159.18 en v2.0) y diversidad de fuentes de datos.

#### 2.1 Conjuntos de Datos para Detección de Objetos Convencionales
Detalla bases de datos de imágenes naturales y destaca que, en imágenes aéreas, el tamaño en megapíxeles y el número de instancias por imagen son métricas más relevantes para comparar escalas que el conteo puro de imágenes.
* **Problemas atacados**: Falta de representatividad de las métricas clásicas de tamaño de imágenes en el dominio de Earth vision.
* **Limitaciones de ese entonces**: Datasets convencionales como PASCAL VOC e ImageNet tienen baja densidad de instancias por imagen y carecen de objetos pequeños aglomerados.
* **Soluciones alcanzadas**: Comparación analítica (Tabla 1) que demuestra que DOTA-v2.0 supera a MS COCO y VOC en el promedio de cajas delimitadoras por imagen (159.18 frente a 7.19 y 2.42).

#### 2.2 Conjuntos de Datos para Detección de Objetos en Imágenes Aéreas
Analiza y tabula las bases de datos aéreas previas detallando su tipo de anotación (HBB, OBB, polígonos, puntos centrales), categorías, número de instancias y sensores.
* **Problemas atacados**: Fragmentación y sesgo de los datasets previos hacia categorías únicas (ej. solo vehículos o barcos).
* **Limitaciones de ese entonces**: Datasets previos eran pequeños, carecían de imágenes de fondo (muestras negativas) o usaban HBBs, lo que no permite la separación de objetos apilados.
* **Soluciones alcanzadas**: Demostración de que DOTA cumple con cuatro principios: volumen sustancial de datos, imágenes grandes con contexto, anotaciones OBB y balance en las fuentes de adquisición.

#### 2.3 Modelos Profundos para Detección de Objetos en Imágenes Aéreas
Discute la evolución de los modelos para manejar rotación (capas invariantes a rotación, filtros rotativos activos ORN, módulos deformables) y variaciones de escala (pirámides de características FPN y pirámides de imágenes), así como la resolución de ambigüedades en la definición del orden de esquinas en cuadriláteros.
* **Problemas atacados**: Variación extrema de orientación y escala, ambigüedad en la definición de la regresión OBB (permutaciones de las esquinas) e insuficiencia de memoria GPU para procesar imágenes gigantescas.
* **Limitaciones de ese entonces**: Los detectores estándar fallan en objetos muy juntos al aplicar NMS horizontal. Además, regresar directamente las esquinas de un cuadrilátero causa inestabilidad numérica por la ambigüedad del punto de inicio.
* **Soluciones alcanzadas**: Clasificación de métodos en basados en regresión de OBB vs. basados en máscaras (segmentación por instancia), y la estrategia estándar de división de imágenes grandes en parches (patches) con posterior fusión mediante NMS orientado.

#### 2.4 Bibliotecas de Código para Detección de Objetos
Revisa el estado de frameworks como Detectron y MMDetection.
* **Problemas atacados**: Incompatibilidad de las bibliotecas de visión tradicionales con las operaciones geométricas necesarias para la rotación.
* **Limitaciones de ese entonces**: Ninguna biblioteca principal soportaba detección orientada de forma nativa ni operaciones como RoI Align rotado de manera eficiente.
* **Soluciones alcanzadas**: Enriquecimiento de la biblioteca MMDetection con operadores matemáticos y de GPU optimizados para detección orientada (Rotated RoI Align).

### 3. Construcción de DOTA
Explica el proceso sistemático de recolección de imágenes, selección de categorías y el flujo de anotación manual con control de calidad.

* **Problemas atacados**: Sesgos de sensores y errores de anotación humana en tareas complejas de rotación.
* **Limitaciones de ese entonces**: Los datasets previos sufrían de sesgos de dominio severos debido al uso de un solo tipo de satélite o sensor.
* **Soluciones alcanzadas**: Colección multisensor de Google Earth, Gaofen-2, Jilin-1 y CycloMedia (fotos aéreas de Rotterdam con vistas oblicuas y nadir).

#### 3.1 Colección de Imágenes
Detalla las fuentes de las imágenes y la inclusión de imágenes aéreas oblicuas (ángulo de inclinación de 45°).
* **Problemas atacados**: Variabilidad de sensores, resoluciones y ángulos de cámara en la vida real.
* **Limitaciones de ese entonces**: Falta de imágenes con vistas oblicuas reales y tamaños representativos del flujo de trabajo cartográfico.
* **Soluciones alcanzadas**: Integración de imágenes gigantes (hasta 29,200 x 27,620 píxeles) para acercarse a la distribución de la producción real.

#### 3.2 Selección de Categorías
Justifica las 18 categorías seleccionadas en función de su frecuencia de aparición y valor práctico.
* **Problemas atacados**: Representación de categorías con límites difusos ("stuff") pero de alto valor contextual.
* **Limitaciones de ese entonces**: Datasets anteriores ignoraban el valor de categorías como aeropuertos o puertos, cuyos límites contextuales ayudan a guiar la búsqueda de aviones o barcos.
* **Soluciones alcanzadas**: Inclusión de 18 categorías, integrando objetos móviles (aviones, helicópteros, vehículos) e infraestructura ("stuff") con fronteras razonablemente definibles (puertos, aeropuertos, helipuertos).

#### 3.3 Anotación de Objetos Orientados
Describe el protocolo de anotación de cuadriláteros ordenados mediante la selección de 4 puntos y la forma de resolver la orientación del "frente/cabeza" del objeto.
* **Problemas atacados**: Dificultad e ineficiencia de rotar manualmente cajas horizontales para ajustarlas a objetos orientados. Ambigüedad de la secuencia de esquinas.
* **Limitaciones de ese entonces**: Rotar cajas horizontalmente requiere ajustar 5 parámetros de forma tediosa. La falta de un criterio para el punto inicial de la caja causaba inestabilidades en la pérdida de regresión.
* **Soluciones alcanzadas**: Interfaz de "clics extremos" en las 4 esquinas del objeto. Para categorías con orientación definida (ej. vehículos), el primer clic denota la "cabeza". Para categorías simétricas, se establece por defecto el vértice superior izquierdo. Se usó un proceso iterativo con grupos de anotadores y revisores expertos.

### 4. Propiedades de DOTA
Analiza cuantitativamente la composición de DOTA-v2.0, cubriendo fuentes de datos, resolución (GSD), distribución de orientaciones, tamaños físicos, relaciones de aspecto (AR), densidades de objetos y la evolución cronológica del dataset.

* **Problemas atacados**: Entender las características intrínsecas del dataset para guiar el diseño de redes neuronales (ej. tamaño de anclas, preprocesamiento).
* **Limitaciones de ese entonces**: Falta de caracterización de la densidad de objetos y su GSD en conjuntos de datos previos.
* **Soluciones alcanzadas**: Análisis detallado de las variaciones y propuesta de estrategias de división del dataset.

#### 4.1 Fuentes de Imágenes
Reporta la distribución de área y píxeles de primer plano (foreground) para cada origen de datos (Google Earth vs. Gaofen y Jilin).
* **Problemas atacados**: Sesgo de muestras positivas.
* **Limitaciones de ese entonces**: Los datasets con exceso de muestras positivas tienden a generar falsos positivos elevados en imágenes de fondo realistas.
* **Soluciones alcanzadas**: Inclusión de grandes extensiones de fondo sin objetos en las imágenes de satélite Gaofen y Jilin para simular entornos reales (menor ratio de foreground, ver Tabla 3).

#### 4.2 Información de Resolución Espacial
Presenta el GSD (Ground Sample Distance) promedio de las imágenes (desde 0.1m/píxel en CycloMedia hasta 4.5m/píxel en Google Earth).
* **Problemas atacados**: Pérdida de invariancia de escala por mezcla de diferentes alturas de vuelo y sensores.
* **Limitaciones de ese entonces**: Sin GSD, los modelos no pueden normalizar el tamaño de los objetos o filtrar ruido.
* **Soluciones alcanzadas**: Proveer GSD para el 30% del dataset y sugerir estimadores de aprendizaje para el resto.

#### 4.3 Variaciones en la Orientación de las Instancias
Presenta el histograma de ángulos de los objetos.
* **Problemas atacados**: Modelado de rotación en 360 grados.
* **Limitaciones de ese entonces**: A diferencia de textos o rostros en fotos naturales, las fotos aéreas no tienen un ángulo preferente (gravedad inútil en vista cenital).
* **Soluciones alcanzadas**: Confirmación estadística de que los ángulos de los objetos en DOTA se distribuyen uniformemente en $[-\pi, \pi]$.

#### 4.4 Variaciones en el Tamaño de Píxel de las Instancias
Mide la distribución del tamaño físico en píxeles, clasificándolos en pequeño (10-50 px), mediano (50-300 px) y grande (>300 px).
* **Problemas atacados**: Detección de objetos extremadamente pequeños.
* **Limitaciones de ese entonces**: Los datasets previos ignoraban objetos por debajo de 10 píxeles.
* **Soluciones alcanzadas**: DOTA-v2.0 anota objetos de hasta 10 píxeles, logrando un 77% de instancias pequeñas, un balance ideal para entrenar redes robustas en multiescala.

#### 4.5 Variaciones en la Relación de Aspecto (AR) de las Instancias
Analiza la relación de aspecto tanto de las cajas OBB como de sus contenedores horizontales HBB.
* **Problemas atacados**: Selección del rango de anclas (anchors) para los detectores basados en regiones.
* **Limitaciones de ese entonces**: Anclas diseñadas para imágenes de COCO (relaciones 1:1, 1:2, 2:1) fallan al detectar vehículos largos o barcos que tienen relaciones de aspecto de hasta 1:10 en OBB.
* **Soluciones alcanzadas**: Provisión de curvas de distribución de AR en DOTA para asistir en el diseño de configuraciones de anclas.

#### 4.6 Variaciones en la Densidad de las Instancias en las Imágenes
Mide la distancia al vecino más cercano de la misma categoría.
* **Problemas atacados**: Detección en aglomeraciones críticas.
* **Limitaciones de ese entonces**: Algoritmos NMS tradicionales eliminan detecciones válidas adyacentes si sus cajas horizontales se solapan.
* **Soluciones alcanzadas**: Identificación de los tanques de almacenamiento, barcos y vehículos pequeños como las categorías más densas, impulsando el desarrollo de técnicas como el Rotated NMS.

#### 4.7 Versiones de DOTA
Compara cronológicamente las versiones del dataset (Tabla 5).

##### 4.7.1 DOTA-v1.0
* **Problemas atacados**: Establecimiento de la primera versión del dataset.
* **Limitaciones de ese entonces**: Ausencia de datos para entrenar redes orientadas a gran escala.
* **Soluciones alcanzadas**: 15 categorías, 2,806 imágenes y 188,282 instancias.

##### 4.7.2 DOTA-v1.5
* **Problemas atacados**: Inclusión de objetos de tamaño sub-10 píxeles y nuevas clases.
* **Limitaciones de ese entonces**: DOTA-v1.0 omitía objetos excesivamente pequeños.
* **Soluciones alcanzadas**: Anotación de instancias diminutas y adición de la categoría "grúa de contenedores" (CC), elevando las instancias a 402,089.

##### 4.7.3 DOTA-v2.0
* **Problemas atacados**: Mitigar el sobreajuste y agregar categorías complejas de infraestructura aérea.
* **Limitaciones de ese entonces**: Desbalances y riesgo de sobreajuste por poseer conjuntos de prueba pequeños.
* **Soluciones alcanzadas**: Expansión masiva a 11,268 imágenes, 1,793,658 instancias y 18 categorías (añadiendo "aeropuerto" y "helipuerto"). Se divide el dataset en entrenamiento, validación, test-dev y test-challenge, liberando los datos sin anotación para pruebas a ciegas en su servidor web.

### 5. Benchmarks
Establece la metodología de evaluación y los detalles de implementación bajo un entorno unificado.

* **Problemas atacados**: Evaluaciones injustas de algoritmos debido a diferencias en plataformas, hiperparámetros o hardware.
* **Limitaciones de ese entonces**: Las publicaciones de ODAI reportaban resultados con diferentes librerías y trucos de entrenamiento, imposibilitando la comparación directa.
* **Soluciones alcanzadas**: Implementación de todos los modelos basándose en una versión modificada de MMDetection con el mismo hardware (Tesla V100 GPU) e hiperparámetros básicos.

#### 5.1 Tareas y Métricas de Evaluación
Define las dos tareas principales (detección con HBB y detección con OBB) y el uso de la métrica mAP de PASCAL VOC 07.
* **Problemas atacados**: Evaluación cuantitativa del error de localización rotacional.
* **Limitaciones de ese entonces**: El cálculo del IoU para polígonos convexos orientados es costoso y no estaba estandarizado.
* **Soluciones alcanzadas**: Implementación eficiente en C/Python para calcular el IoU entre cuadriláteros descomponiéndolos en triángulos convexos.

#### 5.2 Detalles de Implementación
Explica el proceso de corte de imágenes gigantes en parches de 1,024 x 1,024 píxeles con un paso de 824 píxeles, el entrenamiento con 4 GPUs (lote de 8 imágenes) y tasas de aprendizaje estandarizadas.
* **Problemas atacados**: Restricciones de memoria en GPUs modernas.
* **Limitaciones de ese entonces**: Reducir la resolución de imágenes satelitales gigantes a un tamaño estándar (ej. 800 px de lado) elimina los objetos de interés.
* **Soluciones alcanzadas**: Técnica de troceado en parches durante entrenamiento e inferencia, mapeando las coordenadas de detección locales de regreso a la imagen global mediante NMS orientado (umbral 0.1 en OBB, 0.3 en HBB).

##### 5.2.1 Líneas Base con HBBs
Detalla los detectores entrenados directamente para cajas horizontales o convertidos desde predicciones orientadas.
* **Problemas atacados**: Establecer el rendimiento de referencia horizontal.
* **Limitaciones de ese entonces**: Comparar OBB con HBB requería conversiones heurísticas no uniformes.
* **Soluciones alcanzadas**: Uso de RetinaNet, Mask R-CNN, Cascade Mask R-CNN, HTC y Faster R-CNN como baselines de HBB.

##### 5.2.2 Líneas Base con OBBs
Describe las adaptaciones realizadas en las cabezas de regresión para predecir $(x, y, w, h, \theta)$.
* **Problemas atacados**: Regresión directa de cajas orientadas con respecto a anclas horizontales.
* **Limitaciones de ese entonces**: La ambigüedad de los ángulos periódicos y el ordenamiento de esquinas generaba gradientes inestables.
* **Soluciones alcanzadas**: Implementación de la regresión de cajas orientadas calculando previamente las 4 formas permutadas del target y seleccionando la de menor distancia euclidiana (Fórmula 1). Se desarrollaron variantes como Faster R-CNN OBB, RetinaNet OBB y Mask R-CNN (adaptando la máscara al OBB mínimo contenedor).

#### 5.3 Base de Código y Kit de Desarrollo
Detalla las herramientas liberadas en GitHub.
* **Problemas atacados**: Dificultades de los investigadores novatos para manejar las tareas de troceado, conversión y evaluación de DOTA.
* **Limitaciones de ese entonces**: Carencia de un kit de desarrollo accesible para ODAI.
* **Soluciones alcanzadas**: Publicación del `DOTA devkit` que incluye: visualización de etiquetas, cálculo acelerado por GPU de IoU en OBB, scripts de evaluación del mAP y utilidades para recortar/fusionar parches.

### 6. Resultados
Presenta y analiza el rendimiento de los 70+ experimentos en precisión, velocidad e impacto de elecciones de diseño.

* **Problemas atacados**: Cuantificar el impacto de los desarrollos recientes en ODAI sobre DOTA y proponer guías prácticas de arquitectura.
* **Limitaciones de ese entonces**: Los resultados de modelos orientados se limitaban a datos sintéticos o datasets muy simples.
* **Soluciones alcanzadas**: Confirmación de que el mAP disminuye drásticamente de DOTA-v1.0 a v2.0 (Tabla 6), confirmando el aumento de dificultad y realismo.

#### 6.1 Resultados del Benchmark y Análisis
Compara los 10 algoritmos basándose en ResNet-50-FPN y mide sus velocidades (FPS) en una GPU V100.
* **Problemas atacados**: Compromiso entre velocidad y precisión en el diseño de detectores aéreos.
* **Limitaciones de ese entonces**: Falta de análisis de eficiencia temporal para modelos OBB.
* **Soluciones alcanzadas**: Faster R-CNN OBB + RoI Transformer se establece como el modelo más robusto, superando a las alternativas en la curva velocidad-precisión (ver Tabla 6).

##### 6.1.1 Mask Head vs. OBB Head
* **Problemas atacados**: Determinar la mejor estrategia de modelado: regresión directa (OBB Head) vs. segmentación de instancias (Mask Head).
* **Limitaciones de ese entonces**: Los métodos basados en máscaras se asumían superiores pero no se habían comparado en velocidad e inferencia OBB directa.
* **Soluciones alcanzadas**: Mask R-CNN supera a Faster R-CNN H-OBB por 0.57 puntos en mAP en DOTA-v2.0, pero a costa de ser 4 FPS más lento. Las cabezas de máscara convergen más rápido pero requieren más FLOPS.

##### 6.1.2 RoI Transformer vs. Deformable RoI Pooling
* **Problemas atacados**: Modelado específico del alineamiento geométrico de las características de la región orientada.
* **Limitaciones de ese entonces**: Módulos generales como Deformable RoI Pooling (Dpool) se asumían suficientes para cualquier deformación geométrica.
* **Soluciones alcanzadas**: El uso de RoI Transformer supera sistemáticamente a Dpool (Tabla 6), demostrando que la supervisión directa de la rotación y el aprendizaje de transformaciones espaciales específicas de la región son cruciales en imágenes aéreas.

##### 6.1.3 Exclusión de Instancias Pequeñas
* **Problemas atacados**: Evitar la inestabilidad numérica provocada por gradientes extremos de objetos diminutos de menos de 10 píxeles.
* **Limitaciones de ese entonces**: Intentar entrenar con muestras demasiado pequeñas causaba explosión de gradientes en OBB.
* **Soluciones alcanzadas**: Evaluar umbrales de filtrado (área $\le$ 50/80, lado $\le$ 10/12). Se comprobó que filtrar estas instancias diminutas tiene un impacto marginal en el mAP global (Tabla 10).

##### 6.1.4 Número de Propuestas (Proposals)
* **Problemas atacados**: Determinar el número óptimo de propuestas de región generadas por la RPN.
* **Limitaciones de ese entonces**: Los detectores de imágenes naturales usan 300 propuestas (óptimo para VOC/COCO).
* **Soluciones alcanzadas**: El mAP mejora drásticamente al incrementar las propuestas hasta 8,000 en DOTA (mejora de 2.2 puntos en RoI Transformer, ver Tabla 11), confirmando la alta densidad de objetos y la necesidad de buscar más candidatos, aunque a expensas de reducir los FPS.

##### 6.1.5 Aumento de Datos (Data Augmentation)
* **Problemas atacados**: Robustez de los modelos ante rotaciones continuas y variaciones de escala del mundo real.
* **Limitaciones de ese entonces**: Los detectores fallan al procesar escalas u orientaciones no vistas en el entrenamiento.
* **Soluciones alcanzadas**: El aumento multiescala y por rotación conjunta en entrenamiento/test provee mejoras masivas en el mAP (Tabla 12, aumento de 65.03% a 77.60% de mAP). Esto demuestra que los operadores de red (FPN, RoI Transformer) no resuelven por completo el problema del modelado de escala y rotación por sí mismos.

##### 6.1.6 Resultados por Clase
* **Problemas atacados**: Evaluar el rendimiento específico de las clases difíciles.
* **Limitaciones de ese entonces**: Desconocimiento de qué clases sufren más con la escala pequeña o aglomeración.
* **Soluciones alcanzadas**: Comparar DOTA-v1.0 con DOTA-v1.5 muestra que la detección de vehículos pequeños (SV) cae de 77.45 a 52.05 de AP debido a la inclusión de objetos sub-10 píxeles. También confirma que los detectores OBB superan holgadamente a los HBB en categorías densas (ej. barcos y vehículos grandes).

##### 6.1.7 Visualización de Resultados
* **Problemas atacados**: Identificar errores recurrentes de los mejores modelos.
* **Limitaciones de ese entonces**: Dificultad para entender cualitativamente las fallas de los detectores.
* **Soluciones alcanzadas**: El análisis cualitativo (Figura 12) revela: 1) RetinaNet OBB tiene menor precisión de localización por desalineamiento de características, 2) los objetos alargados y auto-similares (como puentes) generan detecciones duplicadas fragmentadas, y 3) existe alta confusión mutua entre puentes, puertos y aeropuertos por sus patrones de textura similares.

#### 6.2 Resultados del Estado del Arte en DOTA-v1.0
* **Problemas atacados**: Comparar el baseline de RoI Transformer con otros métodos publicados en la literatura.
* **Limitaciones de ese entonces**: Falta de comparaciones con aumentos de datos estándar.
* **Soluciones alcanzadas**: Faster R-CNN OBB + RoI Transformer con aumentos alcanza un mAP de 79.82% en OBB, superando a métodos reconocidos como $S^2A$-Net, Gliding Vertex y SCRDet.

#### 6.3 Resultados del Desafío DOAI 2019
* **Problemas atacados**: Rendimiento de los modelos en competiciones abiertas.
* **Limitaciones de ese entonces**: Las soluciones de los retos solían estar sobre-ajustadas o depender de ensambles complejos de modelos.
* **Soluciones alcanzadas**: Muestra que los ganadores del reto del CVPR 2019 (como USTC-NELSLIP o pca lab) utilizaron ensambles masivos, mientras que el modelo de un solo paso propuesto por los autores (Faster R-CNN OBB + RT) alcanzó 76.43% de mAP en el OBB Task, convirtiéndose en el mejor modelo individual reportado.

### 7. Conclusión
* **Problemas atacados**: Resumen de los logros y líneas futuras.
* **Limitaciones de ese entonces**: Falta de herramientas estandarizadas y bases sólidas para el dominio de Earth vision.
* **Soluciones alcanzadas**: Publicación de DOTA-v2.0 como benchmark crucial, demostrando que las reglas de diseño para imágenes naturales no se aplican directamente a imágenes aéreas, abriendo preguntas teóricas para la detección universal de objetos.
