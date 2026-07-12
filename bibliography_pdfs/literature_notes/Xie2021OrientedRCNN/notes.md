# Oriented R-CNN for Object Detection

- **Key**: Xie2021OrientedRCNN
- **Year**: 2021
- **Venue**: ICCV

## Resumen
Oriented R-CNN es un framework general de detección de objetos orientados de dos etapas en imágenes, diseñado para superar el cuello de botella computacional que representa la generación de propuestas de regiones orientadas (oriented proposals) en detectores tradicionales. En la primera etapa, introduce un módulo liviano llamado **Oriented Region Proposal Network (Oriented RPN)**, el cual genera propuestas orientadas de alta calidad de forma casi gratuita a nivel computacional (1/3000 de los parámetros de RoI Transformer+ y 1/15 de Rotated RPN). Esto se logra mediante una novedosa **representación de compensación del punto medio (midpoint offset representation)** que requiere solo 6 parámetros de regresión. En la segunda etapa, el **Oriented R-CNN Head** refina y clasifica las propuestas orientadas tras extraer características invariantes a la rotación mediante **Rotated RoIAlign**. Oriented R-CNN logra una precisión sobresaliente en DOTA (75.87% mAP) y HRSC2016 (96.50% mAP) a una velocidad de 15.1 FPS en una sola GPU RTX 2080Ti, demostrando ser un modelo muy balanceado en precisión y eficiencia.

## Secciones y Subsecciones

### 1. Introduction
Esta sección presenta la motivación del artículo, resaltando la ineficiencia de los métodos actuales de dos etapas en la generación de propuestas de regiones orientadas y planteando la posibilidad de diseñar un Oriented RPN directo y de bajo costo.
* **Problemas atacados**: El alto costo computacional y la ineficiencia temporal asociados a la generación de propuestas de regiones orientadas en detectores de objetos orientados de dos etapas, lo que frena su uso práctico en tareas en tiempo real.
* **Limitaciones de ese entonces**: 
  * Los métodos como *Rotated RPN* colocan densamente anchors rotados con múltiples escalas, relaciones de aspecto y ángulos (por ejemplo, 54 anchors por ubicación), generando un consumo excesivo de memoria y cómputo.
  * Módulos como *RoI Transformer* aprenden propuestas orientadas a partir de RoIs horizontales mediante un proceso secuencial complejo que involucra RPN, RoI Alignment y regresión, elevando drásticamente el costo computacional.
* **Soluciones alcanzadas**: Se propone *Oriented R-CNN*, un detector de dos etapas con un *Oriented RPN* extremadamente ligero que genera propuestas orientadas de alta calidad modificando la rama de regresión de RPN estándar para predecir 6 parámetros (midpoint offset representation) en lugar de 4.

### 2. Related Work
Se realiza una revisión de los avances previos en detección de objetos genéricos y orientados, clasificándolos en métodos basados en propuestas de dos etapas y métodos de una sola etapa o libres de anchors (anchor-free).
* **Problemas atacados**: La desalineación entre las características extraídas y los objetos orientados debido al uso de bounding boxes horizontales tradicionales (que incluyen excesivo ruido de fondo).
* **Limitaciones de ese entonces**:
  * Los detectores basados en anchors rotados tienen problemas de explosión de cómputo.
  * Trabajos basados en *gliding vertex* o *RoI Transformer* añaden capas completamente conectadas densas o alineamientos complejos que resultan lentos.
  * Los detectores de una sola etapa (como *S2ANet* o *R3Det*) intentan solucionar la desalineación mediante alineamiento de características en una sola etapa, pero sus regresiones y asignaciones dinámicas de anchors suelen ser complejas y difíciles de optimizar.
* **Soluciones alcanzadas**: El framework propuesto se define como un detector de dos etapas robusto que preserva las ventajas de la arquitectura R-CNN, eliminando el cuello de botella de la generación de propuestas al proponer un Oriented RPN directo y liviano.

### 3. Oriented R-CNN
Se detalla la arquitectura del framework, la cual se compone de un extractor de características ResNet-FPN que genera cinco niveles de mapas de características y alimenta tanto al Oriented RPN como a la cabeza del detector.
* **Problemas atacados**: Integración ineficiente y desarticulada del ángulo de rotación en la primera etapa de detección de un pipeline de dos etapas.
* **Limitaciones de ese entonces**: Los detectores tradicionales de dos etapas no generan propuestas orientadas directamente en el RPN inicial sin incurrir en un costo computacional y de memoria muy alto.
* **Soluciones alcanzadas**: Se propone un pipeline extremo a extremo que procesa la imagen a través de FPN para producir mapas de características, genera propuestas orientadas eficientes mediante el *Oriented RPN* y realiza la clasificación y regresión poligonal final en el *Oriented R-CNN Head*.

#### 3.1. Oriented RPN
Describe el diseño específico del Oriented RPN que procesa los mapas de características FPN para producir propuestas orientadas en forma de paralelogramos.
* **Problemas atacados**: La necesidad de generar propuestas orientadas directamente sin usar anchors rotados o transformaciones de alineación pesadas en la primera etapa del modelo.
* **Limitaciones de ese entonces**: RPN estándar solo produce propuestas de cajas horizontales (4 parámetros), lo que requiere adaptaciones complejas posteriores para detectar orientaciones.
* **Soluciones alcanzadas**: Se introduce un Oriented RPN que coloca 3 anchors horizontales por ubicación y amplía la rama de regresión a 6 parámetros. Al decodificar estos parámetros mediante offsets de puntos medios, se obtienen propuestas orientadas en forma de paralelogramo.

#### 3.1.1. Midpoint Offset Representation
Introduce la formulación matemática de la representación por compensación del punto medio para objetos con orientaciones arbitrarias.
* **Problemas atacados**: Inestabilidades geométricas y problemas de discontinuidad angular que se presentan en la regresión directa del ángulo ($\theta$) o coordenadas de los vértices.
* **Limitaciones de ese entonces**: La representación basada en ángulos sufre el problema de frontera (boundary problem) debido a la periodicidad del ángulo, y la regresión directa de los cuatro vértices no garantiza que la figura resultante mantenga una forma regular de paralelogramo.
* **Soluciones alcanzadas**: Se utiliza un vector de 6 parámetros $O = (x, y, w, h, \Delta\alpha, \Delta\beta)$, donde $(x, y)$ define el centro, $w$ y $h$ la anchura y altura del rectángulo horizontal delimitador externo, y $\Delta\alpha$ y $\Delta\beta$ representan los desplazamientos de los vértices con respecto a los puntos medios de las aristas superior y derecha. Esta representación hereda el mecanismo de regresión horizontal y ofrece restricciones acotadas y estables para las propuestas.

#### 3.1.2. Loss Function
Se detalla la función de pérdida multitarea utilizada para entrenar el Oriented RPN.
* **Problemas atacados**: Optimización conjunta de la clasificación de objetos (objectness) y la regresión geométrica de las cajas orientadas.
* **Limitaciones de ese entonces**: Dificultades de convergencia de las funciones de pérdida al mezclar regresión de distancias lineales y desplazamientos angulares sin una normalización adecuada.
* **Soluciones alcanzadas**: Se define una pérdida multitarea que suma una pérdida de clasificación por entropía cruzada y una pérdida de regresión utilizando Smooth L1 aplicada a las diferencias normalizadas de los 6 parámetros del midpoint offset representation.

#### 3.2. Oriented R-CNN Head
Describe la segunda etapa del framework orientada a la clasificación final y refinamiento de las propuestas.
* **Problemas atacados**: Clasificación inexacta y desalineación de características de las propuestas que provienen de la primera etapa.
* **Limitaciones de ese entonces**: El uso de propuestas horizontales o paralelogramos mal proyectados en la cabeza de clasificación diluye las características del objeto y reduce la precisión final.
* **Soluciones alcanzadas**: Se procesan las propuestas orientadas a través de un módulo de extracción de características invariante a la rotación y se clasifican/refinan mediante dos capas completamente conectadas (FC) e independientes.

#### 3.2.1. Rotated RoIAlign
Detalla el proceso de alineación de regiones rotadas para extraer vectores de características fijos.
* **Problemas atacados**: Extracción de características a partir de propuestas orientadas de tipo paralelogramo (forma irregular) para ingresarlas a capas completamente conectadas de entrada fija.
* **Limitaciones de ese entonces**: RoIAlign estándar solo maneja áreas horizontales, lo que hace que las características rotadas se distorsionen o capturen ruido de fondo.
* **Soluciones alcanzadas**: El paralelogramo de la propuesta orientada se ajusta a un rectángulo orientado extendiendo la diagonal menor para que iguale a la diagonal mayor. Luego, se realiza una proyección y se divide la región en cuadrículas de $m \times m$ aplicando una transformación de rotación matemática para obtener un mapa de características alineado de tamaño fijo.

#### 3.3. Implementation Details
Explica la configuración del entrenamiento y el proceso de post-procesamiento durante la inferencia.
* **Problemas atacados**: La acumulación de propuestas altamente superpuestas que ralentiza la inferencia en la segunda etapa.
* **Limitaciones de ese entonces**: Realizar NMS poligonal en miles de propuestas orientadas directamente es costoso y consume demasiado tiempo de cómputo.
* **Soluciones alcanzadas**: Se retienen 2000 propuestas por nivel de FPN, se aplica un NMS horizontal veloz con umbral de 0.8 y se seleccionan las 1000 mejores para la cabeza del detector. En la segunda etapa, se aplica un NMS poligonal (poly NMS) por clase con un umbral de IoU de 0.1.

### 4. Experiments
Se presentan los experimentos y comparaciones cuantitativas del framework en dos conjuntos de datos de referencia: DOTA y HRSC2016.
* **Problemas atacados**: Validación de la precisión, velocidad y escalabilidad de la propuesta frente a otros algoritmos del estado del arte.
* **Limitaciones de ese entonces**: Dificultad para comparar equitativamente la eficiencia temporal (FPS) y precisión (mAP) en el mismo entorno de hardware.
* **Soluciones alcanzadas**: Se realizan comparaciones con múltiples métodos demostrando que Oriented R-CNN supera a todos los modelos comparados en precisión, con una velocidad equiparable a modelos de una sola etapa.

#### 4.1. Datasets
Detalla los conjuntos de datos en los que se evaluó el algoritmo.
* **Problemas atacados**: Medir el desempeño en conjuntos de datos aéreos y satelitales con escalas variadas e imágenes de grandes dimensiones.
* **Limitaciones de ese entonces**: DOTA contiene imágenes gigantescas (800x800 a 4000x4000 px) con objetos muy densos y en orientaciones diversas. HRSC2016 se enfoca en barcos con relaciones de aspecto extremas.
* **Soluciones alcanzadas**: DOTA se procesa recortando imágenes en parches de 1024x1024 con solapamiento de 200 px. HRSC2016 se utiliza en su resolución nativa redimensionando el lado menor a 800 píxeles.

#### 4.2. Parameter settings
Configuración de hiperparámetros de entrenamiento e inferencia.
* **Problemas atacados**: Asegurar la estabilidad del entrenamiento y la reproducibilidad de los resultados.
* **Limitaciones de ese entonces**: Diferencias en los cronogramas de entrenamiento (learning rate schedules) y backbones de los experimentos previos.
* **Soluciones alcanzadas**: Uso de optimizador SGD con momento 0.9 y weight decay 0.0001. Entrenamiento por 12 épocas en DOTA (LR inicial de 0.005 disminuyendo en las épocas 8 y 11) y 36 épocas en HRSC2016.

#### 4.3. Evaluation of Oriented RPN
Analiza individualmente el rendimiento del Oriented RPN propuesto.
* **Problemas atacados**: Medir si el Oriented RPN liviano y simplificado logra mantener un recall adecuado de objetos orientados.
* **Limitaciones de ese entonces**: Los detectores livianos corren el riesgo de perder objetos pequeños o con ángulos extremos si el número de propuestas se reduce.
* **Soluciones alcanzadas**: La evaluación muestra que Oriented RPN alcanza un 92.80% de recall con 2000 propuestas y mantiene un sólido 92.20% con 1000 propuestas, justificando este último número para acelerar el procesamiento.

#### 4.4. Comparison with State-of-the-Arts
Compara los resultados finales en términos de mAP.
* **Problemas atacados**: Comparación justa con otros detectores consolidados en la literatura.
* **Limitaciones de ese entonces**: Métodos anteriores sacrifican demasiada precisión para lograr velocidad o viceversa.
* **Soluciones alcanzadas**: Oriented R-CNN alcanza 75.87% mAP en DOTA y 96.50% mAP en HRSC2016 con ResNet50, superando incluso a modelos que usan ResNet101 y confirmando su eficacia.

#### 4.5. Speed versus Accuracy
Estudio comparativo de la velocidad (FPS) y la precisión.
* **Problemas atacados**: Cuello de botella temporal de los detectores de dos etapas en comparación con los de una sola etapa.
* **Limitaciones de ese entonces**: Los detectores de una sola etapa son rápidos pero carecen de precisión fina, mientras que los de dos etapas son muy lentos.
* **Soluciones alcanzadas**: Oriented R-CNN opera a 15.1 FPS en una sola GPU RTX 2080Ti, una velocidad casi idéntica al detector de una sola etapa S2ANet (15.3 FPS) pero ofreciendo un mAP significativamente superior.

### 5. Conclusions
Resumen final del trabajo y líneas de desarrollo futuro.
* **Problemas atacados**: Definir un detector robusto y listo para implementaciones prácticas de visión por computadora en imágenes satelitales.
* **Limitaciones de ese entonces**: Muchos detectores son teóricamente precisos pero inviables en producción debido a su lentitud computacional.
* **Soluciones alcanzadas**: El framework propuesto demuestra que es posible combinar un RPN orientado de bajo costo y una cabeza precisa de dos etapas, estableciendo una nueva e importante línea base (baseline) en detección de objetos rotados.
