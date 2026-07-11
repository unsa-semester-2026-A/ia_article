# Deep Learning-Based Computer Vision Methods for Complex Traffic Environments Perception: A Review

- **Key**: Azfar2024ComplexTrafficReview
- **Year**: 2024
- **Venue**: Data Science for Transportation (Springer)

## Resumen
Esta revisión exhaustiva examina el uso de métodos de visión por computadora basados en aprendizaje profundo (Deep Learning, DL) para la percepción en entornos de tráfico complejos, enfocándose en sistemas de transporte inteligente (ITS) y conducción autónoma (AD). El artículo clasifica y discute detalladamente los desafíos prácticos que surgen al trasladar los modelos del laboratorio al mundo real, agrupándolos en desafíos de datos, desafíos del modelo y desafíos de entornos urbanos complejos. Asimismo, detalla aplicaciones clave como estimación de flujo, detección de congestión, percepción y segmentación en conducción autónoma, percepción cooperativa, interacción de vehículos, predicción de comportamiento de usuarios de la vía, detección de anomalías y computación en el borde (edge computing). Finalmente, propone direcciones futuras de investigación para abordar estas limitaciones.

## Secciones y Subsecciones

### I. Introduction
La introducción presenta la transición de las aplicaciones de visión por computadora en ITS y AD hacia arquitecturas de redes neuronales profundas (DNN). Explica que a pesar del gran rendimiento en benchmarks de laboratorio, la implementación en el mundo real se topa con brechas significativas debido a la variabilidad del entorno y las limitaciones de hardware.
* **Problemas atacados**: La brecha de rendimiento entre los entornos de laboratorio controlados y las condiciones complejas del mundo real para la percepción del tráfico.
* **Limitaciones de ese entonces**: Los algoritmos tradicionales dependían de características hechas a mano (hand-crafted features) que no podían generalizarse ante variaciones de iluminación, clima o tipos de vehículos.
* **Soluciones alcanzadas**: La adopción generalizada de técnicas de aprendizaje profundo que extraen características semánticas robustas de forma automática, aunque introducen nuevos retos de almacenamiento, cómputo y transferencia que esta revisión analiza sistemáticamente.

### II. Data Challenges
Esta sección discute los desafíos relacionados con la recolección, transmisión, calidad y privacidad de los datos en sistemas de tráfico basados en visión.
* **Problemas atacados**: La dificultad de recopilar y utilizar datos representativos y de alta calidad para entrenar y evaluar modelos de visión en transporte.
* **Limitaciones de ese entonces**: Los datasets eran limitados en tamaño, no representaban eventos raros pero críticos (accidentes) y presentaban sesgos geográficos o de clase significativos.
* **Soluciones alcanzadas**: El uso combinado de técnicas de generación de datos sintéticos, aprendizaje activo, federado y arquitecturas en el borde para mejorar el volumen de entrenamiento sin comprometer la privacidad.

#### A. Data communication
Analiza los problemas de latencia y pérdida de paquetes al transmitir transmisiones de video HD a servidores centrales.
* **Problemas atacados**: La alta latencia y la inestabilidad en la transferencia de datos en tiempo real necesarios para la toma de decisiones críticas en ITS.
* **Limitaciones de ese entonces**: El ancho de banda móvil e inalámbrico clásico no permitía la transmisión continua de múltiples flujos de video 4K sin pérdida de paquetes o retrasos significativos.
* **Soluciones alcanzadas**: Desarrollo de protocolos de compresión adaptativos, segmentación inteligente de datos y el auge del procesamiento híbrido borde-nube para minimizar el volumen de datos transmitidos.

#### B. Quality of training data and benchmarks
Revisa la escasez de datos sobre colisiones o incidentes viales raros y la falta de benchmarks realistas.
* **Problemas atacados**: La falta de datos reales etiquetados de situaciones extremas (near-misses o colisiones) esenciales para garantizar la seguridad.
* **Limitaciones de ese entonces**: Capturar colisiones viales en vivo es extremadamente inusual, y recrearlas físicamente es peligroso y costoso.
* **Soluciones alcanzadas**: Utilización de entornos virtuales y simuladores (como CARLA o GTA-V) para generar datos sintéticos realistas, complementados con técnicas de adaptación de dominio para minimizar la discrepancia entre simulación y realidad.

#### C. Data bias
Describe la distribución de cola larga (long-tail) común en los datasets de tráfico reales.
* **Problemas atacados**: El sesgo de predicción de los modelos que tienden a ignorar categorías minoritarias pero críticas (ej. peatones en silla de ruedas, ciclistas, camiones de basura).
* **Limitaciones de ese entonces**: Los métodos convencionales de remuestreo (oversampling) provocaban sobreajuste (overfitting) en clases raras, y las funciones de pérdida estándar no ponderaban adecuadamente el desbalance.
* **Soluciones alcanzadas**: Implementación de pérdidas balanceadas (como la pérdida focal o pérdidas basadas en margen), aumento de datos sintéticos dirigidos a clases raras y técnicas de aprendizaje con pocos ejemplos (few-shot learning).

#### D. High data volume
Examina el reto de computar la inmensa cantidad de datos viales disponibles.
* **Problemas atacados**: La saturación de la capacidad de procesamiento local y de red ante flujos ininterrumpidos de video urbano.
* **Limitaciones de ese entonces**: Los sistemas requerían costosos clústeres de servidores en la nube para procesar datos agregados, resultando inviables a gran escala.
* **Soluciones alcanzadas**: Despliegue de arquitecturas de filtrado inteligente de cuadros y procesamiento local en el borde (edge) que descartan cuadros redundantes.

#### E. Security and Privacy
Discute la privacidad de transeúntes y conductores.
* **Problemas atacados**: El riesgo legal y ético de violar la privacidad individual a través de la captura masiva de rostros y matrículas de vehículos.
* **Limitaciones de ese entonces**: Anonimizar manualmente millones de cuadros era imposible y los algoritmos automáticos antiguos fallaban ante cambios de ángulo o desenfoque.
* **Soluciones alcanzadas**: Desarrollo de algoritmos de deidentificación y difuminado en tiempo real integrados a nivel del propio sensor de cámara (edge node) antes de transmitir la información.

### III. Model Challenges
Aborda las dificultades inherentes al diseño, optimización, robustez y explicación de los modelos basados en aprendizaje profundo.
* **Problemas atacados**: La complejidad computacional y la opacidad en el comportamiento de las redes neuronales profundas.
* **Limitaciones de ese entonces**: Los modelos eran lentos, consumían recursos excesivos y se comportaban como cajas negras, impidiendo su uso en entornos regulados de seguridad crítica.
* **Soluciones alcanzadas**: Aplicación de técnicas de compresión de modelos, herramientas de visualización de características y validación en entornos reales bajo simulación de ruido.

#### A. Complexity
Se enfoca en el tamaño y consumo computacional de los modelos.
* **Problemas atacados**: La imposibilidad de ejecutar modelos pesados del estado del arte (como grandes transformadores o arquitecturas multi-etapa) en hardware embebido en el vehículo.
* **Limitaciones de ese entonces**: La miniaturización de chips no avanzaba tan rápido como la demanda computacional de las nuevas redes neuronales.
* **Soluciones alcanzadas**: Uso de técnicas de destilación de conocimiento, poda de canales redundantes (pruning), y cuantización de precisión (de FP32 a INT8) para acelerar la inferencia en chips embebidos.

#### B. Lack of explainability
Discute la naturaleza de caja negra de las DNNs.
* **Problemas atacados**: La falta de confianza y la imposibilidad de auditar las decisiones automatizadas de conducción ante accidentes inesperados.
* **Limitaciones de ese entonces**: No existían formas robustas de rastrear qué características visuales provocaron una predicción errónea o una acción peligrosa.
* **Soluciones alcanzadas**: Métodos de IA explicable (XAI) como mapas de activación (CAM), mapas de saliencia y análisis de atención en Transformers que permiten rastrear los píxeles más influyentes en la predicción.

#### C. Transferability and generalizability
Trata la sensibilidad de los modelos ante variaciones de dominio.
* **Problemas atacados**: La drástica pérdida de precisión de un detector de tráfico entrenado en una ciudad cuando se despliega en otra región geográfica con diferente infraestructura o señalización.
* **Limitaciones de ese entonces**: Los modelos se entrenaban asumiendo datos idénticamente distribuidos ($i.i.d.$) y no contaban con mecanismos de adaptación.
* **Soluciones alcanzadas**: Uso de aprendizaje auto-supervisado, meta-aprendizaje y redes adversarias de dominio (como CycleGAN) para forzar a la red a extraer características invariantes al dominio.

#### D. Real-world testing
Analiza la robustez de los modelos ante escenarios no estructurados.
* **Problemas atacados**: La inestabilidad de las predicciones en vivo frente a pequeñas perturbaciones de ruido o comportamientos de peatones que no siguen las reglas.
* **Limitaciones de ese entonces**: Los modelos se probaban solo en conjuntos de prueba estáticos de laboratorio, encubriendo problemas de subespecificación operativa.
* **Soluciones alcanzadas**: Introducción de pruebas de robustez sintética (como COCO-C) y el desarrollo de simuladores interactivos de bucle cerrado donde el comportamiento del modelo altera el entorno.

### IV. Complex Traffic Environments
Explica las condiciones físicas y urbanas difíciles de modelar en la visión de tráfico.
* **Problemas atacados**: El deterioro de la señal de imagen y las oclusiones visuales en el mundo físico.
* **Limitaciones de ese entonces**: Los detectores tradicionales fallaban al perder la pista visual del objeto debido a variaciones abruptas en el ambiente.
* **Soluciones alcanzadas**: Fusión de datos temporales, preprocesamiento de eliminación de ruido por clima y arquitecturas de atención espacial.

#### A. Shadow, lighting, weather
Examina la degradación de imagen por factores ambientales.
* **Problemas atacados**: Falsos negativos y fallas de segmentación bajo lluvia intensa, nieve, niebla o contrastes extremos de sombras.
* **Limitaciones de ese entonces**: Los algoritmos dependían de umbrales fijos de contraste e histogramas de color que se distorsionaban ante la niebla o la noche.
* **Soluciones alcanzadas**: Desarrollo de subredes de desneblinado (dehazing), eliminación de lluvia (deraining) basadas en redes generativas y el uso de imágenes térmicas o fusión de sensores.

#### B. Occlusion
Discute la pérdida de visibilidad de los objetos.
* **Problemas atacados**: Pérdida de identidad y desconexión en el seguimiento (tracking) de peatones y autos cuando pasan detrás de postes u otros vehículos.
* **Limitaciones de ese entonces**: Los algoritmos de tracking confiaban demasiado en la superposición de cajas y fallaban cuando los objetos permanecían ocluidos por más de unos pocos cuadros.
* **Soluciones alcanzadas**: Introducción de filtros de Kalman avanzados, modelos de atención temporal (LSTM, Transformers) que retienen la memoria del objeto, y predicción probabilística de trayectorias.

#### C. Camera angle
Discute la variabilidad de la perspectiva de las cámaras viales.
* **Problemas atacados**: Errores en la estimación de escala y orientación del objeto debido a la distorsión por la perspectiva de la cámara.
* **Limitaciones de ese entonces**: Las redes necesitaban ser reentrenadas específicamente para cada nuevo ángulo de cámara en el que se desplegaban.
* **Soluciones alcanzadas**: Diseño de modelos con estimación de homografía tridimensional implícita y algoritmos de proyección inversa que mapean características de imagen a un plano común 2D de coordenadas de carretera.

#### D. Camera blur and degraded images
Aborda las imperfecciones en el lente de la cámara.
* **Problemas atacados**: Desenfoque por movimiento del vehículo o vibración inducida por el viento en los postes de infraestructura.
* **Limitaciones de ese entonces**: Las imágenes borrosas atenuaban los bordes de alta frecuencia que los detectores utilizaban para clasificar los objetos.
* **Soluciones alcanzadas**: Algoritmos de restauración de imágenes en tiempo real y aumento de datos sintético mediante la inyección deliberada de desenfoque y aberraciones ópticas durante el entrenamiento.

#### E. Heterogeneous, urban traffic conditions
Estudia el desorden del tráfico urbano en ciudades congestionadas.
* **Problemas atacados**: La complejidad de modelar la interacción desordenada entre peatones, vehículos no motorizados y autos compartiendo la misma calle.
* **Limitaciones de ese entonces**: Los modelos asumían carriles estructurados y comportamientos estrictamente conformes a las reglas de tráfico ideales.
* **Soluciones alcanzadas**: Uso de redes de grafos para representar dinámicamente las relaciones espaciales y de vecindad de los actores de la vía, capturando interacciones complejas.

### V. Applications
Describe las aplicaciones clave de visión por computadora en el ámbito ITS y conducción autónoma.
* **Problemas atacados**: La integración e implementación final de técnicas de visión en servicios y aplicaciones operativas del mundo real.
* **Limitaciones de ese entonces**: Las aplicaciones se desarrollaban como silos aislados, con una baja integración temporal o espacial y altos costos de cómputo por tarea.
* **Soluciones alcanzadas**: Consolidación de pipelines multitarea que integran detección, segmentación y tracking en arquitecturas optimizadas para procesadores de borde.

#### A. Traffic flow estimation
Trata la estimación y monitoreo del volumen de tráfico.
* **Problemas atacados**: El conteo preciso de vehículos y la estimación de velocidad promedio en avenidas urbanas.
* **Limitaciones de ese entonces**: Métodos anteriores dependían de sensores inductivos físicos en el asfalto (costosos de mantener) o procesamiento manual de video en salas de control.
* **Soluciones alcanzadas**: Integración de detectores de una sola etapa (ej. YOLOv3/v4) combinados con algoritmos de tracking eficientes como DeepSORT y SORT aplicados en tiempo real sobre CCTV.

#### B. Traffic congestion detection
Se enfoca en la identificación automática de embotellamientos viales.
* **Problemas atacados**: La demora en reportar cuellos de botella y vehículos averiados en autopistas.
* **Limitaciones de ese entonces**: Clasificaciones basadas en imágenes globales daban falsos positivos ante autos estacionados o tráfico lento normal.
* **Soluciones alcanzadas**: Algoritmos híbridos que calculan mapas de densidad y velocidad espacial de las cajas delimitadoras, clasificando el nivel de congestión mediante regresión temporal robusta.

#### C. Autonomous driving perception: detection
Detección de objetos viales esenciales (señales, marcas, semáforos, obstáculos).
* **Problemas atacados**: La detección precisa de marcas de carril y señalizaciones en entornos de conducción dinámica.
* **Limitaciones de ese entonces**: Los detectores tradicionales fallaban al oscurecerse la vía o deteriorarse la pintura de los carriles.
* **Soluciones alcanzadas**: Redes de segmentación de líneas (ej. LaneATT, U-Net) optimizadas para la noche mediante preprocesamiento de iluminación adaptativa a nivel de píxel y adaptación de dominio.

#### D. Autonomous driving perception: segmentation
Segmentación detallada de escenas (semántica, de instancias y panóptica).
* **Problemas atacados**: La delimitación precisa de fronteras de aceras, zonas transitables y siluetas individuales de objetos.
* **Limitaciones de ese entonces**: La segmentación semántica clásica era extremadamente lenta para aplicaciones de conducción activa en tiempo real.
* **Soluciones alcanzadas**: Arquitecturas de segmentación de instancias rápidas de una sola etapa (ej. YOLACT) y redes que refinan las características en pirámides eficientes (Dense-RefineDet).

#### E. Cooperative perception
Estudia la colaboración sensorial vehículo-a-todo (V2X).
* **Problemas atacados**: Los puntos ciegos causados por oclusiones severas en intersecciones concurridas.
* **Limitaciones de ese entonces**: Los vehículos solo disponían de sus sensores locales a bordo, limitando su horizonte visual ante vehículos pesados estacionados al frente.
* **Soluciones alcanzadas**: Esquemas de percepción cooperativa que fusionan características intermedias (fusión intermedia con DiscoNet o F-Cooper) transmitidas mediante canales inalámbricos optimizados (V2V/V2I).

#### F. Vehicle interaction
Analiza el modelado de riesgos y colisiones inminentes.
* **Problemas atacados**: Identificar situaciones de riesgo de colisión en intersecciones de forma preventiva.
* **Limitaciones de ese entonces**: Las alarmas de colisión se basaban únicamente en distancias de radar y daban múltiples falsas alarmas en tráfico denso normal.
* **Soluciones alcanzadas**: Implementación de redes neuronales recurrentes con atención temporal (ej. DSA) que procesan el flujo óptico de cámaras frontales para inferir la probabilidad de colisión inminente.

#### G. Road user behavior prediction
Predicción de la trayectoria futura de vehículos y peatones.
* **Problemas atacados**: Anticipar si un peatón en la acera decidirá cruzar intempestivamente la calle.
* **Limitaciones de ese entonces**: Modelos físicos simples de velocidad constante fallaban ante giros rápidos o cambios de dirección repentinos de los usuarios.
* **Soluciones alcanzadas**: Redes híbridas LSTM-CNN y Transformers de trayectoria que modelan el contexto histórico del peatón y las interacciones espaciales de los actores circundantes.

#### H. Traffic anomaly detection
Detección automática de incidentes anómalos (dirección contraria, objetos caídos).
* **Problemas atacados**: Reportar de forma inmediata incidentes inusuales en autopistas para evitar carambolas o accidentes mayores.
* **Limitaciones de ese entonces**: Los operadores humanos en centros de control no pueden monitorear cientos de cámaras simultáneamente sin fatiga.
* **Soluciones alcanzadas**: Algoritmos de aprendizaje no supervisado basados en autoencoders convolucionales 3D y LSTMs que aprenden el patrón normal del tráfico valles y detectan desviaciones severas como anomalías.

#### I. Edge computing
Despliegue del procesamiento cerca del origen de la información.
* **Problemas atacados**: El costo computacional inmanejable y la latencia excesiva asociados a procesar flujos de sensores brutos en nubes remotas.
* **Limitaciones de ese entonces**: Procesadores embebidos antiguos no contaban con aceleración de hardware para inferencia neuronal.
* **Soluciones alcanzadas**: Redes de computación en el borde distribuidas en Unidades de Carretera (Roadside Units - RSUs) y vehículos, integrando compresión de modelos y técnicas de aprendizaje federado.

### VI. Future Directions
Propone las rutas de investigación futuras recomendadas por los autores para mitigar las limitaciones remanentes.
* **Problemas atacados**: La necesidad de establecer soluciones sostenibles a largo plazo para la generalización y costo del aprendizaje profundo en ITS.
* **Limitaciones de ese entonces**: Los métodos siguen adoleciendo de una gran dependencia del etiquetado manual y presentan fragilidad ante entornos imprevistos no cubiertos en el entrenamiento.
* **Soluciones alcanzadas**: Enfoques futuros basados en modelos fundacionales de transporte, etiquetado automatizado asistido por IA, percepción multimodal fusionada y bucles de aprendizaje continuo sobre el terreno.

#### A. For solving data challenges
Rutas para la escasez de datos etiquetados de calidad.
* **Problemas atacados**: La ineficiencia y el costo de etiquetar manualmente petabytes de video de tráfico diario.
* **Limitaciones de ese entonces**: El etiquetado manual genera errores humanos y consume meses de trabajo para un dataset mediano.
* **Soluciones alcanzadas**: Fomento del aprendizaje auto-supervisado, simulaciones sintéticas fotorrealistas de bucle cerrado generadas por motores gráficos modernos y modelos fundacionales preentrenados.

#### B. For solving model challenges
Direcciones para simplificar y dar robustez a las redes.
* **Problemas atacados**: El despliegue a gran escala de modelos en procesadores de bajo costo y la falta de adaptabilidad una vez desplegados.
* **Limitaciones de ese entonces**: Una vez que el modelo se graba en el chip del vehículo, este permanece estático y no aprende de los nuevos errores que experimenta.
* **Soluciones alcanzadas**: Investigación en aprendizaje en línea continuo (online continual learning), bucles de retroalimentación activa y arquitecturas de atención eficientes adaptables a restricciones dinámicas de hardware.

#### C. For solving complex traffic environment challenges
Rutas para los problemas físicos extremos del ambiente urbano.
* **Problemas atacados**: Fallos catastróficos en visión ante tormentas de nieve, lluvia torrencial o falta total de visibilidad.
* **Limitaciones de ese entonces**: Confiar exclusivamente en sensores de espectro visible (cámaras RGB) limita fundamentalmente la capacidad del sistema ante mala visibilidad física.
* **Soluciones alcanzadas**: Fomento de la percepción cooperativa multimodal integrada (Cámaras + LiDAR + Radares + Audio de tráfico) combinada con Transformers de espacio-tiempo para asimilar dinámicas complejas.

### VII. Conclusions
La sección final resume las tesis del trabajo, enfatizando que cerrar la brecha entre el laboratorio y la realidad es el reto más importante.
* **Problemas atacados**: La integración coordinada y segura de la IA en la infraestructura de transporte global.
* **Limitaciones de ese entonces**: Los modelos tienden a evaluarse de forma aislada sin considerar la robustez, privacidad y latencia en entornos reales complejos.
* **Soluciones alcanzadas**: Se concluye que para habilitar una autonomía real (ITS y AD), el desarrollo de IA debe balancear la precisión de validación con la resiliencia ambiental, la explicabilidad, la privacidad de datos y la viabilidad del cómputo en el borde.
