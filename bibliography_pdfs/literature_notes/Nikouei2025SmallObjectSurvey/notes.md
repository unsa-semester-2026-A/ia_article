# Small object detection: A comprehensive survey on challenges, techniques and real-world applications

- **Key**: Nikouei2025SmallObjectSurvey
- **Year**: 2025
- **Venue**: Intelligent Systems with Applications

## Resumen
Este artículo presenta una revisión exhaustiva y actualizada sobre la detección de objetos pequeños (Small Object Detection, SOD) mediante técnicas de aprendizaje profundo, centrándose especialmente en publicaciones científicas de alta calidad (revistas Q1) del período 2024-2025. Los autores analizan los desafíos fundamentales del área (baja resolución, oclusión, ruido de fondo y desbalance de clases), detallan las definiciones métricas (absolutas y relativas), y categorizan las principales soluciones tecnológicas: optimización de redes ligeras, fusión de características multinivel, mecanismos de atención espacial/canal, super-resolución y aumento de datos sintéticos. Además, revisan los conjuntos de datos de referencia y discuten aplicaciones prácticas críticas en agricultura de precisión, vigilancia marítima y aérea, e inspección industrial de defectos.

## Secciones y Subsecciones

### 1- Introduction
Introduce la relevancia de SOD en la visión artificial moderna, donde pequeños detalles marcan diferencias de seguridad crítica (ej., detección de peatones en conducción autónoma o anomalías médicas). Justifica esta encuesta basada en la necesidad de consolidar la literatura Q1 más reciente (2024-2025) y describe la estructura general del trabajo.

### 2- Definitions and Background
Examina los criterios para catalogar un objeto como "pequeño":

#### 2-1- Pixel-Based Definitions
Definición absoluta basada en la cantidad de píxeles (ej., el estándar de MS COCO que define objetos pequeños como aquellos menores a $32 \times 32$ píxeles, o en satélites menores a $20 \times 20$ píxeles).

#### 2-2- Relative Size Criteria
Definición basada en la proporción del objeto respecto al tamaño total de la imagen (usualmente ocupando menos del 1% del área).

#### 2-3- Categorization of Small Objects
Clasifica los objetos en subgrupos como Tiny Objects (áreas menores a 4500 píxeles cuadrados) y Dense Small Objects (DSO, objetos pequeños y densamente agrupados en escenas congestionadas).

#### 2-4- Challenges Highlighted by Definitions
Discute cómo estas definiciones numéricas se traducen en problemas de baja cantidad de píxeles informativos, susceptibilidad al ruido de fondo y alta variabilidad de escala.

### 3- Challenges in SOD
Analiza en profundidad los problemas técnicos de la disciplina:

#### 3-1- Limited Appearance Information and Occlusion
La ausencia de bordes nítidos, texturas o colores distintivos, sumada a la vulnerabilidad a factores climáticos o polvo en entornos industriales.

#### 3-2- Challenges in Localization and Scale Variation
La dificultad para lograr valores altos de Intersection-over-Union (IoU) con cajas de anclaje (anchor boxes) y problemas de alineación del campo receptivo.

#### 3-3- Inefficiency in Feature Learning and Background Interference
La pérdida de información fina durante el submuestreo (downsampling) en redes convolucionales y la baja relación señal-ruido (SNR) provocada por fondos complejos.

#### 3-4- Limitations of Popular Detection Methods
El sesgo de los detectores convencionales (optimizados para objetos medianos/grandes) y el desbalance de tareas en el aprendizaje multitarea.

#### 3-5- High Computational Costs and Hardware Resource Limitations
La restricción de procesamiento en dispositivos embebidos en el borde (Edge computing, UAVs, robots).

#### 3-6- Inconsistent Performance Across Different Scales and Datasets
Las brechas de precisión entre fases de entrenamiento y prueba debido a variaciones extremas en las escalas de los objetos.

#### 3-7- Solutions and Emerging Approaches
Perspectivas como MgD (Multi-Granularity Detection) y FPNs optimizadas para salvar la brecha semántica entre capas.

### 4- Deep Learning Techniques for SOD
Revisa las metodologías punteras del estado del arte:

#### 4-1- Recent Trends
Muestra la distribución del enfoque de investigación actual: optimización de backbones (23.1%), atención (18.5%), extracción de características (16.9%), fusión multi-escala (15.4%), aprendizaje avanzado (13.8%), y alineación de alta resolución (12.3%).

##### 4-1-1- Model Optimization and Lightweight Architectures
Modelos ligeros y eficientes como FFEDet y KDSMALL para plataformas móviles.

##### 4-1-2- Feature Fusion and Multi-Scale Detection
Integración de Transformers y kernels grandes para robustecer la representación espacial.

##### 4-1-3- Anchor-Free and Transformer-Based Approaches
El abandono de anclas fijas en favor de predicciones directas y mecanismos de auto-atención.

##### 4-1-4- Advanced Data Augmentation Techniques
Generación sintética con GANs y modelos de difusión para simular escenarios variados y raros.

##### 4-1-5- Integration of Multi-Modal and Multi-Domain Data
Fusión de datos visuales, térmicos, LiDAR e infrarrojos.

##### 4-1-6- Knowledge Distillation and Self-Supervised Learning
Destilación de conocimiento (KD) docente-estudiante para transferir precisión a modelos ligeros.

#### 4-2- Neural Network Architecture
Analiza innovaciones concretas como la red SOPANet (con residuales y pérdida SOIoU), ScorePillar (para nubes de puntos de LiDAR de baja densidad), y LA-YOLO (con Bi-AFPN-P2 neck y DD-Head para defectos en aisladores).

#### 4-3- Clarity and Visual Information Improvement
Estrategias para incrementar la nitidez visual:

#### 4-3-1- SR for Image Quality Enhancement
Super-resolución (SR) mediante redes adversarias generativas para recuperar texturas perdidas.

#### 4-3-2- Utilizing Multi-Scale Information
Uso de pirámides multinivel que fusionan detalles geométricos de capas superficiales con semántica profunda.

#### 4-3-3- Fusion of Information Across Network Layers
Fusión de capas de red mediante concatenación o módulos de atención (ej., SAM y CAM).

#### 4-4- Data Augmentation and Synthetic Data
Técnicas de entrenamiento robusto:

#### 4-4-1- Data Augmentation Techniques for SOD
Métodos específicos (rotación, recorte contextual, Mosaic, MixUp y CutMix) para balancear la presencia de objetos pequeños.

#### 4-4-2- Synthetic Data Generation for SOD
Uso de renderizado 3D (Blender/CARLA) y redes generativas para mitigar el desbalance de clases de manera automática.

#### 4-4-3- Advantages of Data Augmentation and Synthetic Data in SOD
Beneficios en generalización, robustez, y eficiencia de costos al evitar el etiquetado manual.

#### 4-4-4- Challenges and Research Directions
El problema de la brecha de dominio (domain gap) entre datos sintéticos y reales y el riesgo de sobre-aumento.

#### 4-5- Multi-task and Transfer Learning
Aprendizaje cruzado para compensar la escasez de muestras:

#### 4-5-1- Multi-Task Learning for SOD
Entrenamiento conjunto de detección con tareas auxiliares (como detección de bordes y estimación de profundidad).

#### 4-5-2- Transfer Learning for SOD
Fine-tuning de modelos entrenados en ImageNet/COCO, y uso de KD para optimizar el rendimiento del modelo en dispositivos del borde.

#### 4-5-3- Integration of Multi-Task and Transfer Learning
Aplicaciones combinadas en imágenes médicas, aéreas e inspección de defectos en líneas de transmisión.

### 5- Datasets and Evaluation Metrics
Describe los recursos para benchmarking de algoritmos:

#### 5-1- Datasets
Analiza conjuntos de datos en tres áreas:
- **Aéreos/Satélite:** VisDrone, DIOR, DOTA, VEDAI, UAVDT, SODA-D y DroneCrowd.
- **Conducción Autónoma:** BDD100K y KITTI.
- **Especializados en SOD:** TinyPerson, WSODD, AI-TOD y datasets sintéticos de desenfoque de movimiento.

#### 5-2- Evaluation Metrics
Describe métricas globales (mAP, Precision, Recall, F1) y específicas para objetos pequeños como $\text{AP}_s$ (de MS COCO para objetos $<1024$ píxeles cuadrados) y métricas de eficiencia temporal (FPS, FLOPs, recuento de parámetros).

### 6- Applications and Real-world Use Cases
Revisa el despliegue práctico en sectores del mundo real:
- **6-1 Remote Sensing and Aerial Surveillance:** Clasificación de vehículos e instalaciones en mapas satelitales.
- **6-2 UAV and Drone Applications:** Búsqueda y rescate en áreas de desastre y monitoreo urbano.
- **6-3 Autonomous Systems:** Conducción en carreteras y barcos autónomos.
- **6-4 Industrial Applications:** Defectos en obleas de silicio, grietas microscópicas e inspección de líneas de alta tensión.
- **6-5 Surveillance and Security:** Seguridad ciudadana ante objetos abandonados o amenazas de drones.
- **6-6 Environmental and Natural Resource Monitoring:** Conteo de fauna y detección de incendios forestales.
- **6-7 Medical Imaging and Biological Analysis:** Detección de microtumores y células anormales.
- **6-8 Maritime and Oceanographic Applications:** Detección de boyas y pequeñas embarcaciones en imágenes de sonar o satélite.
- **6-9 Intelligent Transportation Systems:** Conteo de vehículos en intersecciones congestionadas.
- **6-10 Agriculture and Forestry:** Conteo de frutas y detección de plagas tempranas.

### 7- Future Directions
Delinea la hoja de ruta investigativa: redes ultraligeras, Transformers híbridos, funciones de pérdida espacialmente ponderadas, integración de información de flujo temporal/vídeo, y la optimización de hardware mediante cuantización adaptativa para el borde.

### 8- Conclusion
Resume que SOD es una subdisciplina madura con amplio potencial industrial, donde los mayores avances futuros provienen del cierre de la brecha de dominio en datos sintéticos, la mejora de los mecanismos de fusión jerárquica y el despliegue eficiente en tiempo real sobre hardware embebido.
