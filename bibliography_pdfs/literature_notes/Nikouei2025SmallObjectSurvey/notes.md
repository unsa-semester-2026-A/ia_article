# Small object detection: A comprehensive survey on challenges, techniques and real-world applications

- **Key**: Nikouei2025SmallObjectSurvey
- **Year**: 2025
- **Venue**: Intelligent Systems with Applications

## Resumen

Este artículo presenta una revisión exhaustiva y actualizada sobre la detección de objetos pequeños (Small Object Detection, SOD) mediante técnicas de aprendizaje profundo, centrándose especialmente en publicaciones científicas de alta calidad (revistas Q1) del período 2024-2025. Los autores analizan los desafíos fundamentales del área (baja resolución, oclusión, ruido de fondo y desbalance de clases), detallan las definiciones métricas (absolutas y relativas), y categorizan las principales soluciones tecnológicas: optimización de redes ligeras, fusión de características multinivel, mecanismos de atención espacial/canal, super-resolución y aumento de datos sintéticos. Además, revisan los conjuntos de datos de referencia y discuten aplicaciones prácticas críticas en agricultura de precisión, vigilancia marítima y aérea, e inspección industrial de defectos.

## Secciones y Subsecciones

### 1- Introduction
Introduce la relevancia de SOD en la visión artificial moderna, donde pequeños detalles marcan diferencias de seguridad crítica (ej., detección de peatones en conducción autónoma o anomalías médicas). Justifica esta encuesta basada en la necesidad de consolidar la literatura Q1 más reciente (2024-2025) y describe la estructura general del trabajo.
* **Problemas atacados**: Falta de robustez en detectores generales al aplicarse a objetos pequeños; consecuencias críticas de fallos en conducción autónoma, medicina y vigilancia.
* **Limitaciones de ese entonces**: Los detectores convencionales optimizados para objetos grandes pierden detalles de objetos de pequeña escala al no estar diseñados para su resolución y características.
* **Soluciones alcanzadas**: Revisión detallada de retos y avances recientes (2024-2025) en SOD basados en aprendizaje profundo, analizando arquitecturas, aumentación y modelos ligeros.

### 2- Definitions and Background
Examina los criterios para catalogar un objeto como "pequeño".
* **Problemas atacados**: Ambigüedad en la definición de "objeto pequeño" y falta de consistencia entre dominios de aplicación.
* **Limitaciones de ese entonces**: Definiciones empíricas que varían entre conjuntos de datos urbanos, satelitales y biológicos, dificultando la comparación.
* **Soluciones alcanzadas**: Clasificación de definiciones en absolutas basadas en píxeles y relativas basadas en área de imagen, estructurando subcategorías como Tiny y DSO.

#### 2-1- Pixel-Based Definitions
Definición absoluta basada en la cantidad de píxeles.
* **Problemas atacados**: Definir un umbral cuantitativo absoluto para catalogar objetos pequeños en imágenes de sensores diversos.
* **Limitaciones de ese entonces**: El estándar de COCO (<32x32 píxeles) no es adecuado para teledetección o satélites donde los objetos suelen ser menores a 20x20 píxeles.
* **Soluciones alcanzadas**: Estandarización de umbrales absolutos adaptados al dominio (e.g., <32x32 para COCO/drones, <20x20 para satélites).

#### 2-2- Relative Size Criteria
Definición basada en la proporción del objeto respecto al tamaño total de la imagen.
* **Problemas atacados**: Objetos con suficiente resolución que aún son pequeños respecto a la resolución total de la imagen (e.g., en imágenes 4K).
* **Limitaciones de ese entonces**: Los umbrales absolutos fallan al evaluar imágenes de ultra alta resolución donde los objetos pequeños superan los 32x32 píxeles de forma absoluta.
* **Soluciones alcanzadas**: Adopción del criterio de área relativa, catalogando como pequeños a objetos que ocupan menos del 1% (o 1.5% en TT-100K) de la imagen.

#### 2-3- Categorization of Small Objects
Clasifica los objetos en subgrupos específicos.
* **Problemas atacados**: La categoría general "small" agrupa objetos con características visuales y distribuciones muy dispares.
* **Limitaciones de ese entonces**: Tratar igual a un objeto pequeño aislado que a objetos densos y congestionados lleva a fallos de detección.
* **Soluciones alcanzadas**: Subcategorización fina en Tiny Objects (áreas menores a 4500 píxeles cuadrados) y Dense Small Objects (DSO) para escenarios congestionados.

#### 2-4- Challenges Highlighted by Definitions
Discute cómo las definiciones numéricas se traducen en problemas prácticos.
* **Problemas atacados**: Relacionar las definiciones cuantitativas con las dificultades reales de percepción de los detectores.
* **Limitaciones de ese entonces**: Las métricas no explican por qué el modelo falla ante cambios de perspectiva o ruido de fondo.
* **Soluciones alcanzadas**: Identificación de retos clave (bajo conteo de píxeles, interferencia de fondo, variabilidad de escala) derivados de las propiedades físicas y geométricas de los objetos pequeños.

### 3- Challenges in SOD
Analiza en profundidad los problemas técnicos de la disciplina.
* **Problemas atacados**: Caracterización sistemática de los retos inherentes que degradan la precisión de SOD.
* **Limitaciones de ese entonces**: Los detectores genéricos no abordan de manera directa factores como la oclusión extrema o la baja relación señal-ruido.
* **Soluciones alcanzadas**: Compilación exhaustiva de los desafíos y conceptualización de una taxonomía de problemas (información limitada, localización, aprendizaje de características, costo de cómputo).

#### 3-1- Limited Appearance Information and Occlusion
La ausencia de bordes nítidos, texturas o colores distintivos, sumada a la oclusión.
* **Problemas atacados**: Falta de rasgos visuales como bordes, texturas o colores nítidos en objetos pequeños.
* **Limitaciones de ese entonces**: El polvo, vapor de agua (e.g., en minas de carbón) y la oclusión física borran por completo la señal del objeto.
* **Soluciones alcanzadas**: Uso de módulos de procesamiento de movimiento (MPM) y alineación de movimiento-visión para guiar la detección en secuencias de video e imágenes degradadas.

#### 3-2- Challenges in Localization and Scale Variation
La dificultad para lograr valores altos de IoU con cajas de anclaje (anchor boxes).
* **Problemas atacados**: Dificultad para obtener IoU alto con cajas delimitadoras fijas y desalineación del campo receptivo por variaciones de escala.
* **Limitaciones de ese entonces**: El fenómeno de concept drift y la latencia del cómputo en la nube para procesamiento en tiempo real.
* **Soluciones alcanzadas**: Adopción de asignaciones dinámicas y representación espacial fina, combinando cómputo en el borde con aprendizaje incremental.

#### 3-3- Inefficiency in Feature Learning and Background Interference
La pérdida de información fina durante el submuestreo en redes convolucionales.
* **Problemas atacados**: Pérdida drástica de características finas (como bordes y esquinas) tras múltiples capas de downsampling convolucional.
* **Limitaciones de ese entonces**: Las FPN clásicas tienen una tensión insalvable entre capas superficiales (buena localización, mala semántica) y profundas (buena semántica, mala localización).
* **Soluciones alcanzadas**: Introducción de módulos de realce y fusión adaptativa de capas (SOPANet, FAM) que integran semántica sin diluir la señal espacial de objetos pequeños.

#### 3-4- Limitations of Popular Detection Methods
El sesgo de los detectores convencionales optimizados para objetos grandes.
* **Problemas atacados**: Los detectores genéricos basados en CNN/Transformers están sesgados hacia objetos medianos/grandes.
* **Limitaciones de ese entonces**: Tienen una alta demanda de recursos en altas resoluciones y sufren desbalance de tareas en el aprendizaje multitarea.
* **Soluciones alcanzadas**: Adaptaciones estructurales específicas para SOD y optimización de funciones de pérdida (e.g., SOIoU, β-CIoU).

#### 3-5- High Computational Costs and Hardware Resource Limitations
La restricción de procesamiento en dispositivos embebidos en el borde.
* **Problemas atacados**: El elevado coste computacional y de latencia al procesar imágenes de alta resolución en dispositivos del borde (UAVs, USVs).
* **Limitaciones de ese entonces**: Limitada memoria y potencia de cálculo en hardware embebido (e.g. Jetson).
* **Soluciones alcanzadas**: Desarrollo de backbones ligeros (como Faster-C2f, HDPConv) y técnicas de destilación de conocimiento.

#### 3-6- Inconsistent Performance Across Different Scales and Datasets
Las brechas de precisión entre fases de entrenamiento y prueba.
* **Problemas atacados**: Disparidad de rendimiento y pobre generalización cuando la distribución de escala cambia entre entrenamiento y prueba.
* **Limitaciones de ese entonces**: Los objetos pequeños caen fuera de los anchors prefijados y se pierden por completo durante las convoluciones profundas.
* **Soluciones alcanzadas**: Uso de estrategias de parches y asignadores dinámicos que no dependen de anclas rígidas.

#### 3-7- Solutions and Emerging Approaches
Perspectivas emergentes para solventar las limitaciones.
* **Problemas atacados**: Integración y validación de las nuevas aproximaciones para solventar el desbalance de muestras y la pérdida semántica.
* **Limitaciones de ese entonces**: Falta de sinergia entre diferentes paradigmas de mejora visual y de red.
* **Soluciones alcanzadas**: Adopción de MgD (detección multi-granularidad), FPNs adaptativas mejoradas y técnicas híbridas de alineación espacial.

### 4- Deep Learning Techniques for SOD
Revisa las metodologías de aprendizaje profundo del estado del arte.
* **Problemas atacados**: Clasificación y análisis de los métodos de aprendizaje profundo propuestos para mejorar la precisión de SOD.
* **Limitaciones de ese entonces**: Dificultad para estructurar un campo de investigación con crecimiento exponencial.
* **Soluciones alcanzadas**: Taxonomía de 6 pilares de técnicas (optimización de modelos, fusión multi-escala, anchor-free, aumentación avanzada, multi-modalidad, destilación/autosupervisado).

#### 4-1- Recent Trends
Muestra la distribución del enfoque de investigación actual.
* **Problemas atacados**: Identificar las prioridades de investigación más recientes y su distribución temática.
* **Limitaciones de ese entonces**: Falta de datos estadísticos sobre qué soluciones son más investigadas.
* **Soluciones alcanzadas**: Cuantificación de tendencias en 2024-2025, destacando backbones optimizados (23.1%) y atenciones (18.5%).

##### 4-1-1- Model Optimization and Lightweight Architectures
Modelos ligeros y eficientes como FFEDet y KDSMALL.
* **Problemas atacados**: Equilibrar la reducción de parámetros del modelo con la retención de características de objetos pequeños.
* **Limitaciones de ese entonces**: Los modelos ligeros estándar (MobileNet) pierden demasiada información espacial fina.
* **Soluciones alcanzadas**: Modelos ligeros específicos para SOD como FFEDet y KDSMALL que combinan super-resolución y destilación de conocimiento.

##### 4-1-2- Feature Fusion and Multi-Scale Detection
Integración de Transformers y kernels grandes para robustecer la representación espacial.
* **Problemas atacados**: Extracción y fusión robusta de características en presencia de variaciones extremas de escala.
* **Limitaciones de ese entonces**: Las FPN estándar no resuelven la desalineación espacial de objetos pequeños en escenas complejas o degradadas.
* **Soluciones alcanzadas**: Integración de mecanismos de atención jerárquica y transformadores espacio-temporales en la fusión de características.

##### 4-1-3- Anchor-Free and Transformer-Based Approaches
El abandono de anclas fijas en favor de predicciones directas y auto-atención.
* **Problemas atacados**: Evitar la rigidez de los anclajes fijos y modelar dependencias a larga distancia.
* **Limitaciones de ese entonces**: NMS clásico puede eliminar predicciones correctas de objetos pequeños densamente agrupados.
* **Soluciones alcanzadas**: Uso de representaciones sin anclas y modelos basados en DETR con mecanismos de atención deformables.

##### 4-1-4- Advanced Data Augmentation Techniques
Generación sintética con GANs y modelos de difusión.
* **Problemas atacados**: Escasez y desbalance severo de muestras de entrenamiento para SOD.
* **Limitaciones de ese entonces**: Las transformaciones básicas (rotación, color) no simulan variaciones ambientales realistas.
* **Soluciones alcanzadas**: Uso de GANs y modelos de difusión para generar parches y fondos sintéticos realistas específicos para SOD.

##### 4-1-5- Integration of Multi-Modal and Multi-Domain Data
Fusión de datos visuales, térmicos, LiDAR e infrarrojos.
* **Problemas atacados**: Falta de fiabilidad de las imágenes RGB bajo malas condiciones de iluminación o clima.
* **Limitaciones de ese entonces**: Fusión ineficiente o desalineada de sensores ópticos con térmicos, LiDAR o radar.
* **Soluciones alcanzadas**: Redes de fusión multi-modal que alinean características espaciales y espectrales para resaltar firmas de objetos pequeños.

##### 4-1-6- Knowledge Distillation and Self-Supervised Learning
Destilación de conocimiento docente-estudiante para transferir precisión a modelos ligeros.
* **Problemas atacados**: Reducir la dependencia de datos anotados y transferir precisión de modelos pesados a ligeros.
* **Limitaciones de ese entonces**: La destilación clásica no prioriza las regiones de objetos pequeños, perdiendo su señal.
* **Soluciones alcanzadas**: Algoritmos de destilación de conocimiento consciente de la escala (e.g. Cross-Scale Distillation) y pretext-tasks autosupervisadas.

#### 4-2- Neural Network Architecture
Analiza innovaciones concretas como SOPANet, ScorePillar y LA-YOLO.
* **Problemas atacados**: Mapear y comparar las innovaciones de red publicadas recientemente para SOD.
* **Limitaciones de ese entonces**: Cada propuesta utiliza diferentes configuraciones experimentales y backbones dificultando la comparación.
* **Soluciones alcanzadas**: Tabla comparativa detallada de modelos recientes (MICPL, ScorePillar, LA-YOLO, MFFSODNet, etc.) analizando sus componentes de backbone, neck y loss.

#### 4-3- Clarity and Visual Information Improvement
Estrategias para incrementar la nitidez visual.
* **Problemas atacados**: Recuperación de la nitidez visual en objetos lejanos o degradados por ruido físico del sensor.
* **Limitaciones de ese entonces**: La súper resolución aislada consume muchos recursos sin asegurar una mejor clasificación posterior.
* **Soluciones alcanzadas**: Integración conjunta de super-resolución, información multi-escala y fusión intercapa para robustecer el mapa de características.

##### 4-3-1- SR for Image Quality Enhancement
Super-resolución mediante redes adversarias generativas.
* **Problemas atacados**: Pérdida irreversible de información por limitaciones físicas de resolución de la cámara.
* **Limitaciones de ese entonces**: Modelos SR genéricos pueden inventar texturas artificiales que confunden al clasificador.
* **Soluciones alcanzadas**: Super-resolución basada en GANs e interpolaciones con restricciones semánticas orientadas específicamente a la reconstrucción de objetos pequeños.

##### 4-3-2- Utilizing Multi-Scale Information
Uso de pirámides multinivel que fusionan detalles geométricos con semántica profunda.
* **Problemas atacados**: Representar simultáneamente el contexto global y el detalle local del objeto pequeño.
* **Limitaciones de ese entonces**: Pérdida de coordenadas espaciales precisas al avanzar en la pirámide de características.
* **Soluciones alcanzadas**: Estructuras piramidales jerárquicas optimizadas (e.g. SCFPN) que acoplan capas profundas y poco profundas.

##### 4-3-3- Fusion of Information Across Network Layers
Fusión de capas de red mediante concatenación o módulos de atención.
* **Problemas atacados**: Cómo combinar la alta precisión de localización de capas poco profundas con la semántica de capas profundas.
* **Limitaciones de ese entonces**: La suma o concatenación ingenua de características genera redundancia y ruido.
* **Soluciones alcanzadas**: Uso de mecanismos de atención dinámica de canal y espacio (e.g., SAM, CAM, EMA) y redes de fusión de características (FFNs).

#### 4-4- Data Augmentation and Synthetic Data
Técnicas de entrenamiento robusto.
* **Problemas atacados**: Mitigar el sesgo del modelo provocado por la escasez de datos etiquetados de objetos pequeños.
* **Limitaciones de ese entonces**: El etiquetado manual es costoso y propenso a errores en objetos de pocos píxeles.
* **Soluciones alcanzadas**: Enfoque combinado de transformaciones de datos y renderizado sintético para balancear clases.

##### 4-4-1- Data Augmentation Techniques for SOD
Métodos específicos (rotación, recorte, Mosaic, MixUp y CutMix).
* **Problemas atacados**: Ajustar el modelo a variaciones de rotación, escala y oclusión en el mundo real.
* **Limitaciones de ese entonces**: Métodos convencionales pueden recortar el objeto pequeño dejándolo fuera de la imagen.
* **Soluciones alcanzadas**: Técnicas avanzadas como MixUp, CutMix y Mosaic adaptadas a objetos pequeños, además de borrado aleatorio y transformaciones espaciales controladas.

##### 4-4-2- Synthetic Data Generation for SOD
Uso de renderizado 3D y redes generativas.
* **Problemas atacados**: Inviabilidad de recolectar imágenes reales de objetos pequeños en condiciones peligrosas o raras.
* **Limitaciones de ese entonces**: Brecha de dominio (domain gap) entre el aspecto artificial de los datos sintéticos y los entornos reales.
* **Soluciones alcanzadas**: Generación de imágenes sintéticas ultra-realistas mediante motores 3D (CARLA, Unity, Blender) y refinamiento con GANs/modelos de difusión.

##### 4-4-3- Advantages of Data Augmentation and Synthetic Data in SOD
Beneficios en generalización, robustez y eficiencia de costos.
* **Problemas atacados**: Demostrar las ventajas de coste y precisión del uso de datos artificiales en SOD.
* **Limitaciones de ese entonces**: La validación del rendimiento a menudo omite analizar si el modelo es robusto a cambios climáticos reales.
* **Soluciones alcanzadas**: Evidencia de mayor generalización, robustez ambiental y mitigación directa del desbalance crítico de clases a bajo costo.

##### 4-4-4- Challenges and Research Directions
El problema de la brecha de dominio y el riesgo de sobre-aumento.
* **Problemas atacados**: Abordar las debilidades del uso excesivo de augmentación y datos sintéticos.
* **Limitaciones de ese entonces**: Riesgo de colapso por "over-augmentation" y la alta demanda de cómputo para simulaciones 3D.
* **Soluciones alcanzadas**: Desarrollo de pipelines automatizados (AutoAugment) y técnicas de adaptación de dominio para cerrar la brecha de distribución de datos.

#### 4-5- Multi-task and Transfer Learning
Aprendizaje cruzado para compensar la escasez de muestras.
* **Problemas atacados**: Aprovechamiento de conocimientos previos y tareas auxiliares para estabilizar el aprendizaje de SOD.
* **Limitaciones de ese entonces**: Falta de correspondencia entre representaciones de objetos grandes de datasets genéricos y objetos pequeños del dominio objetivo.
* **Soluciones alcanzadas**: Integración de tareas secundarias y transferencia de conocimiento inter-dominios.

##### 4-5-1- Multi-Task Learning for SOD
Entrenamiento conjunto de detección con tareas auxiliares.
* **Problemas atacados**: Entrenar el modelo con múltiples objetivos para guiar indirectamente la localización fina.
* **Limitaciones de ese entonces**: Conflicto entre tareas que degradan el rendimiento de la detección (e.g. segmentación compitiendo con cajas).
* **Soluciones alcanzadas**: Diseño de cabezales específicos compartiendo backbone, y uso de tareas auxiliares como detección de bordes o super-resolución con pesos dinámicos de pérdida.

##### 4-5-2- Transfer Learning for SOD
Fine-tuning de modelos entrenados en ImageNet/COCO, y uso de KD.
* **Problemas atacados**: Ajustar redes pre-entrenadas a conjuntos de datos con escasez severa de anotaciones SOD.
* **Limitaciones de ese entonces**: Domain shift que disminuye la precisión al transferir weights de imágenes terrestres a aéreas.
* **Soluciones alcanzadas**: Fine-tuning por capas, pre-entrenamiento específico de dominio y destilación de conocimiento profesor-alumno.

##### 4-5-3- Integration of Multi-Task and Transfer Learning
Aplicaciones combinadas en imágenes médicas, aéreas e inspección de defectos.
* **Problemas atacados**: Sinergia de ambos enfoques para maximizar la generalización en dominios complejos.
* **Limitaciones de ese entonces**: Incremento de la complejidad del pipeline de entrenamiento y riesgo de olvido catastrófico.
* **Soluciones alcanzadas**: Frameworks unificados aplicados a imágenes médicas (lesiones micro), aéreas e inspección industrial de componentes.

### 5- Datasets and Evaluation Metrics
Describe los recursos para benchmarking de algoritmos.
* **Problemas atacados**: Estructurar los recursos y herramientas de validación para SOD.
* **Limitaciones de ese entonces**: La falta de estandarización en conjuntos de datos y métricas oscurece la comparación objetiva.
* **Soluciones alcanzadas**: Compilación detallada de 13 datasets y discusión de métricas específicas de escala y eficiencia.

#### 5-1- Datasets
Analiza conjuntos de datos en tres áreas principales.
* **Problemas atacados**: Seleccionar el conjunto de datos de entrenamiento adecuado según el escenario operativo.
* **Limitaciones de ese entonces**: Los datasets generales no representan los retos específicos de UAVs, satélites o entornos industriales.
* **Soluciones alcanzadas**: Censo y catalogación de datasets clave organizados por dominio: Aéreos (VisDrone, DIOR, DOTA, VEDAI), UAV-Based (UAVDT, SODA-D, DroneCrowd), Conducción (BDD100K, KITTI), Especializados (TinyPerson, WSODD, AI-TOD) y sintéticos.

#### 5-2- Evaluation Metrics
Describe métricas globales y específicas para objetos pequeños.
* **Problemas atacados**: Medir de forma precisa la calidad de localización y el tiempo de respuesta del modelo en SOD.
* **Limitaciones de ese entonces**: AP general (mAP50) es demasiado permisivo para la precisión de localización requerida en SOD, mientras que métricas muy estrictas ocultan progresos reales.
* **Soluciones alcanzadas**: Adopción de métricas de tamaño específico (AP_S, AP_T), métricas para video (T-AP) y parámetros de hardware (FPS, FLOPs).

### 6- Applications and Real-world Use Cases
Revisa el despliegue práctico en sectores del mundo real.
* **Problemas atacados**: Mapear el impacto práctico y la transferencia tecnológica de SOD a industrias reales.
* **Limitaciones de ese entonces**: Muchos algoritmos con excelente rendimiento teórico son inviables de desplegar en entornos operativos con recursos y hardware restringidos.
* **Soluciones alcanzadas**: Clasificación sistemática en 10 casos de uso del mundo real (vigilancia aérea, UAVs, conducción autónoma, inspección industrial, seguridad, medio ambiente, medicina, marítimo, ITS, agricultura) detallando los beneficios de SOD en cada uno.

### 7- Future Directions
Delinea la hoja de ruta investigativa.
* **Problemas atacados**: Delinear la hoja de ruta científica para resolver los problemas persistentes de SOD.
* **Limitaciones de ese entonces**: Los detectores más precisos siguen siendo pesados y propensos a falsos positivos en entornos dinámicos.
* **Soluciones alcanzadas**: Identificación de 8 líneas futuras clave, incluyendo redes ultraligeras, Transformers híbridos, pérdidas de localización espacialmente ponderadas, e integración temporal en video.

### 8- Conclusion
Resume que SOD es una subdisciplina madura con amplio potencial industrial.
* **Problemas atacados**: Resumir el estado de la técnica y las implicaciones industriales de SOD.
* **Limitaciones de ese entonces**: Persistencia de desafíos de generalización y costo computacional en hardware del borde.
* **Soluciones alcanzadas**: Conclusión que destaca a SOD como una disciplina madura pero en evolución, donde la sinergia de datos sintéticos, modelos comprimidos y multimodalidad representa el futuro del área.
