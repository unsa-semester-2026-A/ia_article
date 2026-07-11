# Context in object detection: a systematic literature review

- **Key**: Jamali2025ContextObjectDetection
- **Year**: 2025
- **Venue**: Artificial Intelligence Review

## Resumen
Este artículo presenta una revisión sistemática de la literatura (Systematic Literature Review, SLR) sobre el uso de la información contextual para mejorar los detectores de objetos. Analiza más de 265 publicaciones de los años 2018 a 2023, evaluando cómo el contexto visual y no visual (e.g., espacial, temporal, semántico, de escala, espectral) ayuda a mitigar retos clásicos como oclusiones, desenfoque por movimiento, variaciones de escala extrema e insuficiencia de datos. El estudio organiza la literatura en siete categorías principales de detección de objetos: general (GOD), en video (VOD), de objetos pequeños (SOD), camuflados (COD), y en configuraciones de pocos ejemplos (ZSD, OSOD, FSOD). Además, clasifica el contexto por niveles (conocimiento previo, local y global) y por tipo de relaciones (por pares o de alto orden), comparando exhaustivamente arquitecturas, backbones, mecanismos de integración y rendimientos cuantitativos (mAP, MAE) en conjuntos de datos estándar (COCO, VOC).

## Secciones y Subsecciones

### 1. Introducción
La detección de objetos en visión artificial se enfrenta a retos persistentes como la similitud interclase, variaciones intraclase, condiciones de iluminación adversas, objetos fuera de su entorno típico (out-of-context) y oclusiones. El uso de contexto (información adicional del entorno, coocurrencia de clases, etc.) ayuda a reducir el espacio de búsqueda espacial y semántica, imitando el sistema visual de los humanos.
* **Problemas atacados**: Baja robustez y fallos de detección en detectores tradicionales y basados en aprendizaje profundo causados por similitud visual de clases, variabilidad de apariencia y condiciones ambientales adversas.
* **Limitaciones de ese entonces**: Los detectores comunes ignoran la coherencia semántica global y las relaciones físicas de coocurrencia de los objetos en una escena, procesando cada candidato de manera aislada.
* **Soluciones alcanzadas**: Presentación de un marco sistemático para entender el impacto del contexto y planteamiento de 5 preguntas de investigación (RQs) enfocadas en el tipo de contexto, integración, backbones, rendimiento comparativo y limitaciones de datos.

### 2. Context
Define el término "contexto" en el ámbito de la visión computacional como cualquier información visual o no visual (posición, textura, firma espectral, escala, tiempo, etc.) que complementa las características locales de un objeto para refinar su clasificación e inferencia espacial.

#### 2.1 The role of context in human vision and computer vision
Estudio comparativo de cómo el cerebro humano asocia elementos visuales en estructuras con sentido (Gestalt) frente a cómo la visión artificial intenta imitar esta capacidad reduciendo el espacio de búsqueda espacial y semántica.
* **Problemas atacados**: Ambigüedad en la identificación de objetos aislados y baja resolución de características locales.
* **Limitaciones de ese entonces**: Los modelos de detección tradicionales evalúan exhaustivamente todas las posiciones y escalas posibles (sliding windows) de forma independiente, lo que resulta ineficiente y propenso a falsos positivos.
* **Soluciones alcanzadas**: Modelar el contexto para guiar la atención espacial hacia regiones probables del objeto y resolver oclusiones mediante información circundante (e.g., inferir la presencia de un teclado por proximidad a una pantalla y mouse).

#### 2.2 Context levels
Clasifica el contexto en tres niveles operativos: conocimiento previo (prior knowledge), contexto local (dentro y alrededor del objeto inmediato) y contexto global (toda la imagen o escena amplia).
* **Problemas atacados**: Incapacidad de los detectores para procesar eficientemente imágenes con niveles de detalle contradictorios o ruidosos.
* **Limitaciones de ese entonces**: El uso exclusivo de contexto local falla en objetos micro, mientras que el uso exclusivo de contexto global puede introducir confusión semántica al mezclar entornos.
* **Soluciones alcanzadas**: Demostración de que la combinación estructurada de representaciones globales (e.g., descriptores Gist) y locales optimiza la discriminación visual de objetos ambiguos.

##### 2.2.1 Contextual interactions in local context
Estudia las interacciones a nivel de pixel (bordes, segmentación fina), región (relación entre parches de imágenes) y objeto (interacciones físicas directas de objetos vecinos).
* **Problemas atacados**: Falta de granularidad en la extracción de relaciones de oclusión y pose local.
* **Limitaciones de ese entonces**: Analizar combinaciones a nivel de píxel requiere un alto costo computacional y tiempo de procesamiento.
* **Soluciones alcanzadas**: Clasificación sistemática de interacciones, sugiriendo que las interacciones a nivel de objeto son computacionalmente menos costosas al simplificar la matriz relacional en función del conteo de instancias detectadas.

##### 2.2.2 Contextual interactions in global context
Analiza cómo el fondo de la escena y las relaciones globales (coocurrencia objeto-escena) guían el refinamiento de hipótesis lógicas.
* **Problemas atacados**: Predicciones fuera de contexto (e.g., barcos en desiertos o trenes en el aire) que degradan el rendimiento de detección.
* **Limitaciones de ese entonces**: Los clasificadores de cajas tradicionales no tienen forma de constreñir las predicciones espaciales según el tipo de entorno de la imagen entera.
* **Soluciones alcanzadas**: Modelado top-down donde la configuración global de la escena sirve de restricción física previa para el detector de objetos.

#### 2.3 Pairwise and higher-order relations
Compara relaciones espaciales entre pares de objetos directos (pairwise, como "sobre", "debajo", "al lado de") y relaciones complejas grupales (higher-order).
* **Problemas atacados**: Modelado de escenas con alta densidad de interacciones dinámicas.
* **Limitaciones de ese entonces**: Limitación de representar interacciones solo como pares independientes, ignorando la estructura contextual grupal (e.g., personas interactuando con herramientas en áreas de trabajo colectivas).
* **Soluciones alcanzadas**: Sistematización del uso de campos aleatorios de Markov (MRF), gráficos condicionales (CRF) y redes neuronales de gráficos (GNN) para inferir relaciones relacionales de alto orden.

#### 2.4 Context types
Categorización detallada de los tipos de contexto según su naturaleza física o informacional: Semántico (coocurrencia lógica), Espacial (geometría, distancias, 2D/3D), Temporal (secuencia de cuadros), De Escala (tamaños proporcionales), Espectral (bandas de luz), y Otros (iluminación, clima, etc.).

##### 2.4.1 Semantic context (probability)
* **Problemas atacados**: Ambigüedad en la categorización de objetos que comparten texturas con otras clases.
* **Limitaciones de ese entonces**: Dependencia exclusiva del detector en características visuales intrínsecas (forma, bordes) que colapsa ante oclusiones.
* **Soluciones alcanzadas**: Uso de matrices de probabilidad de coocurrencia semántica para ajustar las predicciones basadas en la compatibilidad lógica de los objetos de la escena.

##### 2.4.2 Spatial context (position)
* **Problemas atacados**: Falta de concordancia física de las ubicaciones predichas para las cajas delimitadoras.
* **Limitaciones de ese entonces**: Ausencia de restricciones geométricas y de profundidad 3D que validen si un objeto está suspendido o correctamente apoyado.
* **Soluciones alcanzadas**: División espacial en coocurrencias físicas, relaciones 2D (dirección, topológicas, distancia) y relaciones 3D para consolidar estimaciones de layout tridimensional.

##### 2.4.3 Scale context (size)
* **Problemas atacados**: Variaciones de escala extremas que confunden los extractores multiescala estándar.
* **Limitaciones de ese entonces**: Búsquedas multiescala exhaustivas que provocan un alto coste computacional.
* **Soluciones alcanzadas**: Uso de "tamaños familiares" (familiar size) relativos para descartar escalas físicamente imposibles de los objetos en función del tamaño del entorno y de otros objetos clave.

##### 2.4.4 Spectral context
* **Problemas atacados**: Dificultad para discernir objetos terrestres o materiales bajo firmas de color óptico idénticas.
* **Limitaciones de ese entonces**: Las imágenes de espectro visible (RGB) no capturan firmas físicas como el calor o reflectancia de clorofila.
* **Soluciones alcanzadas**: Integración de bandas de infrarrojo cercano (NIR) y reflectancia vegetal/mineral para complementar la detección terrestre.

##### 2.4.5 Spatial-spectral context
* **Problemas atacados**: Desalineación entre firmas espectrales y características de límites espaciales en imágenes satelitales.
* **Limitaciones de ese entonces**: Los detectores procesan las bandas y las texturas espaciales por separado sin agregación contextual integrada.
* **Soluciones alcanzadas**: Frameworks como S2ADet que agregan información espacial y de bandas espectrales simultáneamente para guiar la inferencia remota.

##### 2.4.6 Temporal context (time)
* **Problemas atacados**: Inestabilidad y parpadeo de detección (flickering) a lo largo de cuadros continuos de video.
* **Limitaciones de ese entonces**: Detectores que evalúan cada fotograma como una imagen estática independiente, perdiendo coherencia y constancia temporal.
* **Soluciones alcanzadas**: Clasificación en contextos de corto plazo (convoluciones temporales, flujo óptico) y de largo plazo (LSTMs, bancos de memoria a largo plazo) para propagar la consistencia de predicción en el tiempo.

### 3. Research method
Establece la metodología de la revisión sistemática, detallando bases de datos consultadas (IEEE Xplore, Scopus, Web of Science), rango de años (2018-2023), y criterios booleanos de búsqueda.
* **Problemas atacados**: Falta de rigor científico, duplicidad y sesgo de selección en la literatura previa sobre visión artificial con contexto.
* **Limitaciones de ese entonces**: Revisiones previas eran narrativas e informales, omitiendo criterios de inclusión/exclusión claros o comparaciones rigurosas basadas en mAP.
* **Soluciones alcanzadas**: Implementación del estándar de revisión de Kitchenham, filtrando 265 publicaciones iniciales hasta retener 117 artículos finales válidos con métricas comparables.

### 4. Analysis and discussion

#### 4.1 Datasets
Revisión crítica de las bases de datos utilizadas para entrenar y evaluar modelos contextuales, describiendo MS COCO, PASCAL VOC, Cityscapes, BDD100K, DOTA, WIDER FACE, Caltech Camera Traps, y S-UODAC2020.
* **Problemas atacados**: Sesgos culturales, limitaciones de tamaño y variabilidad de condiciones en los benchmarks de evaluación.
* **Limitaciones de ese entonces**: Dataset clásicos como COCO sesgan a los detectores hacia entornos urbanos occidentales, e ignoran dinámicas específicas (como imágenes submarinas o aéreas con cajas orientadas).
* **Soluciones alcanzadas**: Tabulación detallada de datasets especificando tareas, tipos de datos, clases y volúmenes de imágenes para orientar la validación objetiva.

#### 4.2 General object detection (GOD)
Análisis de siete enfoques de integración contextual para detección genérica de imágenes estáticas: basados en grafos, jerárquicos, aumentación de datos contextuales, multiescala, basados en RPN, basados en atención y otros.
* **Problemas atacados**: Integración deficiente de señales contextuales en arquitecturas clásicas como detectores de una y dos etapas.
* **Limitaciones de ese entonces**: Inserción ad-hoc de bloques de atención que aumentan el costo computacional sin mejoras sustanciales en la precisión del detector.
* **Soluciones alcanzadas**: Clasificación estructurada y tabulación exhaustiva comparativa en COCO, VOC 2007 y VOC 2012 de docenas de modelos, demostrando qué variantes de grafos (e.g., Distilling Knowledge Graph) y jerárquicos (e.g., HCE) dominan sus respectivas categorías.

#### 4.3 Small object detection (SOD)
Revisión de detectores diseñados para objetos de tamaño $\le 32 \times 32$ píxeles mediante el uso de contexto circundante.
* **Problemas atacados**: Ausencia de píxeles informativos en objetos muy lejanos o pequeños en los que el detector carece de características locales.
* **Limitaciones de ese entonces**: El submuestreo de las capas convolucionales reduce a cero las activaciones del objeto micro.
* **Soluciones alcanzadas**: Análisis de 17 modelos específicos (F-SSD, CEFP2N, IENet, mejorados con atención como CGA-YOLO y Eagle-YOLO), demostrando que la integración de pirámides conscientes del contexto (como CAB Net) y campos de sensación expandidos (MCS-YOLOv4) rescatan la señal atenuada de objetos diminutos.

#### 4.4 Video object detection (VOD)
Evaluación de métodos que explotan la información inter-cuadro para estabilizar las detecciones en video.
* **Problemas atacados**: Degradación visual temporal de los objetos debido al desenfoque por movimiento (motion blur) y oclusiones instantáneas.
* **Limitaciones de ese entonces**: Los detectores de una etapa fallan al perder la caja en cuadros borrosos sucesivos.
* **Soluciones alcanzadas**: Modelos como Context Faster R-CNN (con memorias a largo y corto plazo), PTSEFormer (transformador espacio-temporal progresivo), y el uso de "contexto de movimiento" (motion context) basado en mapas de correlación y flujo óptico para guiar el detector.

#### 4.5 Zero-shot, one-shot, and few-shot object detection
Análisis de detectores adaptados para detectar categorías invisibles durante el entrenamiento o basadas en muestras sumamente reducidas.
* **Problemas atacados**: Sobreajuste (overfitting) catastrófico y colapso de características de clases novedosas ante la escasez crítica de anotaciones de entrenamiento.
* **Limitaciones de ese entonces**: Los enfoques de meta-aprendizaje y transferencia clásicos no logran diferenciar objetos ambiguos sin datos masivos.
* **Soluciones alcanzadas**: Uso de contexto semántico global basado en embeddings textuales (SRR-FSD) y transformadores de afinidad de contexto (Context-Transformer) que transfieren el conocimiento relacional de las clases conocidas para orientar la detección de objetos poco frecuentes o invisibles.

#### 4.6 Camouflaged object detection (COD)
Estudio de modelos contextuales para localizar objetos cuyos colores y texturas imitan intencionalmente el fondo.
* **Problemas atacados**: Incapacidad de detectores tradicionales para delimitar objetos que no presentan contraste en sus contornos (e.g., animales camuflados).
* **Limitaciones de ese entonces**: Los detectores tradicionales fallan al no encontrar gradientes fuertes de borde ni disparidad de textura.
* **Soluciones alcanzadas**: Redes de fusión cruzada consciente del contexto (C2F-Net), realce de bordes mediante transformadas de Fourier (BCNet) y estimación de importancia de píxeles (PCPNet), logrando reducir drásticamente el error absoluto medio (MAE) en bases como COD10K.

### 5. Conclusion, research gaps, and limitations
Respuestas definitivas a las RQs y planteamiento de vacíos y limitaciones identificados.
* **Problemas atacados**: Falta de robustez en dominios cruzados, dependencia de contextos estáticos, y ausencia de rigor estadístico (e.g., intervalos de confianza) en la validación de detectores visuales.
* **Limitaciones de ese entonces**: Los modelos actuales no son flexibles a cambios dinámicos del contexto y a menudo ignoran la estimación de la profundidad.
* **Soluciones alcanzadas**: Delineación de 9 vacíos de investigación crítica (como el uso de modelos multimodales con lenguaje VLMs, integración de contexto de escala con conciencia de profundidad, y cuantificación de incertidumbres).
