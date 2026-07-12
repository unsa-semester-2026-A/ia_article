# Deep Learning-Based Computer Vision Methods for Complex Traffic Environments Perception: A Review

- **Key**: Azfar2024ComplexTrafficReview
- **Year**: 2024
- **Venue**: Communications in Transportation Research

## Resumen
Este artículo presenta una revisión bibliográfica exhaustiva de las aplicaciones de visión por computadora basadas en aprendizaje profundo (Deep Learning) para la percepción en sistemas de transporte inteligente (ITS) y conducción autónoma (AD). El estudio examina de forma sistemática los desafíos recurrentes categorizados en tres áreas principales: desafíos de **datos** (comunicación, calidad, sesgo, alto volumen y privacidad), desafíos de **modelos** (complejidad computacional, falta de explicabilidad, transferabilidad y pruebas en el mundo real) y desafíos de los **entornos de tráfico complejos** (sombras, iluminación, clima adverso, oclusión, ángulos de cámara y tráfico urbano heterogéneo). Asimismo, analiza aplicaciones clave como la estimación del flujo de tráfico, la detección de congestión, la percepción en conducción autónoma (detección y segmentación), la percepción cooperativa, la interacción vehicular, la predicción del comportamiento de usuarios viales, la detección de anomalías y la computación en el borde (edge computing). Finalmente, propone directrices y orientaciones futuras para hacer viables y seguros estos sistemas en despliegues reales en el mundo físico.

## Secciones y Subsecciones

### I. Introduction
Establece el contexto de la transición desde la videovigilancia manual y semi-automatizada hacia métodos completamente automatizados basados en redes neuronales profundas (DNN).
* **Problemas atacados**: La necesidad de procesar de manera rápida, confiable y a gran escala los flujos de video de videovigilancia de tráfico para la gestión de incidentes y el control en tiempo real en conducción autónoma.
* **Limitaciones de ese entonces**: A pesar de las mejoras reportadas en datasets de referencia estáticos (benchmarks), existen brechas muy importantes que impiden el despliegue comercial seguro en el mundo real debido a fallas bajo condiciones urbanas no controladas y la inestabilidad de los modelos ante variaciones ambientales.
* **Soluciones alcanzadas**: Clasificar de forma estructurada los retos en datos, modelos y entorno, evaluando soluciones de mitigación existentes y delineando el camino para el desarrollo de modelos adaptables en el borde.

### II. Data Challenges
Analiza los retos asociados con el ciclo de vida de los datos requeridos por los algoritmos de visión por computadora en transporte.

#### A. Data communication
Describe el flujo de comunicación de datos entre sensores e infraestructura de nube o TMC (Traffic Management Center).
* **Problemas atacados**: Cuellos de botella en la transmisión de video rico en datos, latencias de red y pérdida de paquetes.
* **Limitaciones de ese entonces**: Desincronización temporal de relojes entre múltiples sensores y dificultades de calibración espacial en entornos vehiculares cooperativos donde las posiciones relativas cambian constantemente.
* **Soluciones alcanzadas**: Uso de algoritmos de calibración dinámica basados en interpolación temporal y estimación de matrices de transformación homográficas a partir de elementos estáticos del fondo.

#### B. Quality of training data and benchmarks
Discute la escasez de datos etiquetados de calidad para eventos de baja frecuencia como choques o cuasi-colisiones.
* **Problemas atacados**: La recolección costosa y el etiquetado manual lento de millones de imágenes de tráfico real.
* **Limitaciones de ese entonces**: Los datasets comerciales carecen de suficientes muestras de accidentes, situaciones de riesgo extremo y modos de transporte minoritarios (como ciclistas).
* **Soluciones alcanzadas**: Uso de simuladores gráficos en 3D de alta fidelidad y videojuegos para generar de forma automática datos anotados de forma masiva, combinados con técnicas de adaptación de dominio para transferir el conocimiento al mundo real.

#### C. Data bias
Estudia el problema del desbalance de clases o distribución de cola larga (long-tail Zipfian distribution).
* **Problemas atacados**: El sesgo de clasificación en el detector que favorece a las categorías dominantes (e.g., automóviles) frente a las minoritarias (e.g., camiones de construcción, ciclistas).
* **Limitaciones de ese entonces**: Estrategias de sobremuestreo y re-ponderación de pérdida tradicionales requieren divisiones rígidas que dañan la consistencia del entrenamiento en las fronteras de clases y merman el poder de discriminación de clases raras.
* **Soluciones alcanzadas**: Aplicación de transfer learning con pesos preentrenados afinados con datos raros, y aplicación de pérdidas ponderadas dinámicas basadas en la frecuencia local.

#### D. High data volume
Examina el reto del ancho de banda y almacenamiento frente al crecimiento de la red de cámaras de tráfico.
* **Problemas atacados**: La imposibilidad física de transmitir y procesar de manera centralizada en la nube el 100% del video generado por cámaras 4K de carreteras y sensores a bordo.
* **Limitaciones de ese entonces**: Dependencia de servidores centralizados pesados que introducen latencias intolerables para decisiones críticas de tráfico.
* **Soluciones alcanzadas**: Arquitecturas de computación distribuida de nube-borde (vehicle-cloud) utilizando Unidades de Borde en Carretera (RSU) para procesar datos de manera local y cercana a la fuente.

#### E. Security and Privacy
Trata el cumplimiento de la privacidad de los ciudadanos y la seguridad de la información.
* **Problemas atacados**: El riesgo de violar la privacidad al capturar rostros y matrículas de vehículos sin consentimiento.
* **Limitaciones de ese entonces**: Anonimizar o desenfocar datos en tiempo real requiere cómputo adicional y el almacenamiento centralizado sigue expuesto a ciberataques.
* **Soluciones alcanzadas**: Procesamiento en el borde limitando la transmisión del video crudo, y aplicación de algoritmos de des-identificación en tiempo real.

### III. Model Challenges
Analiza las limitaciones estructurales y operativas de los modelos de Deep Learning.

#### A. Complexity
* **Problemas atacados**: Los altos requisitos de cómputo y consumo de energía de las DNN.
* **Limitaciones de ese entonces**: Modelos potentes diseñados para servidores con múltiples GPUs no pueden ejecutarse en tiempo real en hardware embebido vehicular o de carretera (e.g. Jetson Nano) debido a limitaciones de memoria y cómputo.
* **Soluciones alcanzadas**: Técnicas de simplificación como la cuantización de modelos, poda de filtros (pruning), redimensionamiento dinámico del video de entrada y uso de mecanismos eficientes de atención deformable (deformable attention).

#### B. Lack of explainability
* **Problemas atacados**: El comportamiento de "caja negra" de las redes neuronales que dificulta la validación de seguridad.
* **Limitaciones de ese entonces**: Los miles de millones de parámetros del modelo hacen imposible explicar analíticamente el porqué de una decisión de frenado o maniobra en conducción autónoma, lo que impide cumplir con regulaciones de seguridad vial.
* **Soluciones alcanzadas**: Uso de herramientas de visualización de atención en vision transformers (co-attention visualizations) y mapas de activación de clases (CAM) para auditar qué regiones de la imagen influyen en las decisiones.

#### C. Transferability and generalizability
* **Problemas atacados**: Degradación del rendimiento cuando el modelo se enfrenta a datos fuera de la distribución de entrenamiento (domain shift).
* **Limitaciones de ese entonces**: Los modelos asumen una distribución de prueba idéntica a la de entrenamiento, fallando catastróficamente al cambiar de ciudad, país, o ante ligeras variaciones de cámara.
* **Soluciones alcanzadas**: Métodos de alineación de dominio para extraer características invariantes y técnicas masivas de regularización mediante aumento de datos.

#### D. Real-world testing
* **Problemas atacados**: El problema de la subespecificación (underspecification) donde modelos con el mismo mAP en validación fallan de forma aleatoria en el mundo real.
* **Limitaciones de ese entonces**: Variabilidad de rendimiento causada por la semilla aleatoria de inicialización y vulnerabilidad ante perturbaciones del mundo real (ruido inalámbrico, ataques por inyección de láser en LiDAR).
* **Soluciones alcanzadas**: Pruebas basadas en escenarios simulados extremos y desarrollo de modelos de aprendizaje en línea que actualizan sus parámetros en tiempo real ante fallos detectados.

### IV. Complex Traffic Environments
Detalla los factores físicos del entorno que degradan las señales visuales.

#### A. Shadow, lighting, weather
* **Problemas atacados**: Cambios abruptos en las condiciones ópticas debido a sombras, noche, neblina, lluvia o nieve.
* **Limitaciones de ese entonces**: Algoritmos de flujo óptico tradicionales o extracción de fondo (GMM) fallan ante reflejos nocturnos, deslumbramiento por faros en túneles o gotas de agua en la lente.
* **Soluciones alcanzadas**: Entrenamiento con datasets específicos de clima adverso (e.g., DAWN), y uso de redes generativas adversarias (GAN) para traducción de dominio día-noche (Style Transfer).

#### B. Occlusion
* **Problemas atacados**: Pérdida de visibilidad de objetos al ser bloqueados por postes u otros vehículos en tráfico denso.
* **Limitaciones de ese entonces**: Pérdida de tracking e identidad de vehículos al desaparecer temporalmente tras un obstáculo.
* **Soluciones alcanzadas**: Pérdidas de repulsión (repulsion losses) que evitan el desplazamiento del bounding box predicho hacia objetos vecinos y fusión cooperativa de vistas de múltiples cámaras.

#### C. Camera angle
* **Problemas atacados**: Variabilidad geométrica provocada por la posición de montaje de las cámaras de vigilancia.
* **Limitaciones de ese entonces**: Redes entrenadas con cámaras de ángulo alto fallan al procesar vistas bajas de patrullas o tomas lejanas donde el objeto ocupa pocos píxeles.
* **Soluciones alcanzadas**: Mapas de perspectiva e información espacio-temporal integrada para ajustar dinámicamente las escalas de predicción.

#### D. Camera blur and degraded images
* **Problemas atacados**: Suciedad física en lentes, vibración de cámaras por viento y blind spots por insectos.
* **Limitaciones de ese entonces**: Generación de falsas alarmas de colisión por distorsión de la imagen.
* **Soluciones alcanzadas**: Uso de redes de segmentación específicas para imágenes degradadas (Dense-Gram networks) y preprocesamiento de restauración de imagen.

#### E. Heterogeneous, urban traffic conditions
* **Problemas atacados**: Flujo vehicular denso y desordenado en intersecciones urbanas con presencia simultánea de automóviles, autobuses, bicicletas y peatones.
* **Limitaciones de ese entonces**: Comportamientos que violan las reglas de tránsito en ciudades densas de Asia que los simuladores occidentales no logran modelar ni predecir.
* **Soluciones alcanzadas**: Recopilación de datasets específicos de tráfico denso y no regulado (e.g., TRAF) y modelos de grafos para capturar interacciones de agentes heterogéneos.

### V. Applications
Describe los métodos de Deep Learning aplicados a tareas específicas y sus mecanismos de mitigación de desafíos.

#### A. Traffic flow estimation
* **Problemas atacados**: Estimación de velocidad, densidad, volumen y colas en carreteras.
* **Limitaciones de ese entonces**: Algoritmos de una sola etapa (YOLOv3+DeepSORT) o dos etapas (Faster R-CNN) sufren fallas catastróficas ante neblina o ángulos de cámara desfavorables.
* **Soluciones alcanzadas**: Implementación de YOLOv3 optimizado en dispositivos en el borde y uso de datasets multiespectrales o específicos de clima adverso para reducir el sesgo.

#### B. Traffic congestion detection
* **Problemas atacados**: Clasificación del nivel de congestión (ligero, medio, denso).
* **Limitaciones de ese entonces**: Dependencia de la precisión de conteo de vehículos individuales, la cual falla en condiciones de oclusión severa.
* **Soluciones alcanzadas**: Fusión de datos multisensoriales (radar, láser) y uso de transferencia de estilo (Style Transfer GAN) para robustecer la clasificación bajo mal clima.

#### C. Autonomous driving perception: detection
* **Problemas atacados**: Detección de señales, semáforos, marcas de carril, peatones y vehículos.
* **Limitaciones de ese entonces**: Los semáforos lejanos y señales ocupan una fracción diminuta de píxeles, lo que causa altas tasas de falsos negativos.
* **Soluciones alcanzadas**: Arquitecturas de agregación de características multiescala (como FAMN), híbridos de CNN y LSTM para rastreo de carriles y pérdidas de repulsión en oclusiones peatonales en multitudes.

#### D. Autonomous driving perception: segmentation
* **Problemas atacados**: Segmentación semántica, de instancias y panóptica de píxeles en tiempo real.
* **Limitaciones de ese entonces**: Los algoritmos tradicionales de segmentación espacial (como Mask R-CNN) son extremadamente lentos para operaciones en milisegundos en el vehículo.
* **Soluciones alcanzadas**: Modelos rápidos de segmentación en tiempo real basados en atención espacial de una sola etapa (e.g., CenterMask, YOLACT, ESE-Seg que codifica formas explícitas mediante operaciones tensoriales veloces).

#### E. Cooperative perception
* **Problemas atacados**: Percepción colectiva integrando datos de múltiples vehículos y semáforos inteligentes a través de V2X.
* **Limitaciones de ese entonces**: El envío de video crudo satura el canal inalámbrico (cuello de botella de transmisión) y la fusión tardía de predicciones pierde información contextual valiosa.
* **Soluciones alcanzadas**: Fusión intermedia (intermediate fusion) que transmite representaciones neuronales comprimidas mediante transformadores de visión (e.g., V2X-ViT y OPV2V).

#### F. Vehicle interaction
* **Problemas atacados**: Modelado de maniobras y trayectorias considerando el comportamiento de vehículos circundantes.
* **Limitaciones de ese entonces**: La mayoría de modelos asumen condiciones de iluminación óptimas y fallan en cruces ferroviarios o bajo lluvia extrema.
* **Soluciones alcanzadas**: Uso de técnicas de aprendizaje de fondo para adaptar el modelo a faros nocturnos, filtros de ruido para lluvia e integración de CycleGAN para adaptación de dominio.

#### G. Road user behavior prediction
* **Problemas atacados**: Predicción de trayectorias futuras de vehículos y usuarios vulnerables (peatones, ciclistas).
* **Limitaciones de ese entonces**: Los modelos puramente secuenciales (LSTMs) no capturan interacciones espaciales de múltiples agentes heterogéneos y son lentos.
* **Soluciones alcanzadas**: Redes de convolución sobre grafos (GCN) espacio-temporales combinadas con codificadores LSTM (e.g. GRIP) y uso de transformers para modelar interacciones en entornos densos.

#### H. Traffic anomaly detection
* **Problemas atacados**: Detección automática de vehículos detenidos, accidentes y objetos en la vía.
* **Limitaciones de ese entonces**: Dependencia de datos etiquetados de anomalías (que son muy escasos) e inestabilidad por el movimiento físico de la cámara provocado por el viento.
* **Soluciones alcanzadas**: Algoritmos de estabilización de video en preprocesamiento, y desarrollo de autoencoders convolucionales espacio-temporales no supervisados que aprenden la distribución del tráfico "normal" para detectar desviaciones anómalas.

#### I. Edge computing
* **Problemas atacados**: Despliegue físico de modelos de visión en dispositivos locales de almacenamiento y cómputo limitado.
* **Limitaciones de ese entonces**: Modelos sobredimensionados e ineficientes que agotan la batería del hardware embebido o saturan el ancho de banda al subir datos.
* **Soluciones alcanzadas**: Algoritmos de compresión y poda estructural de redes, federated learning para entrenar en el borde preservando la privacidad y compresión de características mediante clustering espectral.

### VI. Future Directions
Delinea las oportunidades y líneas de investigación prioritarias.

#### A. For solving data challenges
* **Problemas atacados**: La baja calidad y escasez de datos etiquetados de esquinas críticas (corner cases).
* **Limitaciones de ese entonces**: Dependencia de etiquetado manual costoso.
* **Soluciones alcanzadas**: Desarrollo de modelos base generalizados (Foundation Models como Florence) que resuelven múltiples tareas aguas abajo, aprendizaje semi-supervisado, crowdsensing y simuladores multimodales (Vista 2.0).

#### B. For solving model challenges
* **Problemas atacados**: Falta de adaptabilidad del modelo a cambios continuos en carretera.
* **Limitaciones de ese entonces**: Estática de los modelos tras su entrenamiento inicial.
* **Soluciones alcanzadas**: Aprendizaje continuo en línea (online learning) en el borde y frameworks de explicabilidad genéricos para modelos espaciotemporales.

#### C. For solving complex traffic environment challenges
* **Problemas atacados**: El ruido visual persistente en entornos metropolitanos complejos.
* **Limitaciones de ese entonces**: Insuficiencia de los sensores ópticos puros en condiciones de clima severo.
* **Soluciones alcanzadas**: Detección multimodal (video + LiDAR + audio para detección de ruidos o accidentes), percepción cooperativa distribuida sobre redes 6G-V2X y transformadores espacio-temporales adaptativos.

### VII. Conclusions
* **Problemas atacados**: Cierre de la revisión científica.
* **Limitaciones de ese entonces**: La degradación del rendimiento de los modelos al enfrentar datos del mundo real.
* **Soluciones alcanzadas**: Confirmar que para la viabilidad de la visión por computadora en ITS, el enfoque debe moverse hacia la robustez ambiental, la explicabilidad, la computación eficiente en el borde y el modelado multimodal de interacciones de agentes.
