# ReDet: A Rotation-equivariant Detector for Aerial Object Detection

- **Key**: Han2021ReDet
- **Year**: 2021
- **Venue**: CVPR

## Resumen
ReDet (Rotation-equivariant Detector) es un detector de objetos en imágenes aéreas que incorpora de manera explícita la equivalencia y la invariancia ante la rotación. A diferencia de los detectores convencionales, que sufren de alta redundancia al codificar orientaciones mediante el aumento masivo de parámetros y datos, ReDet propone una solución estructural en dos partes. Primero, adopta un backbone equivalente a la rotación (**ReResNet con ReFPN**) basado en convoluciones de grupo que reduce los parámetros del extractor de características a $1/N$ (disminuyendo el peso del modelo completo de 313 Mb a 121 Mb, un 60% menos). Segundo, introduce un alineador de regiones invariante a la rotación (**Rotation-invariant RoI Align - RiRoI Align**), el cual extrae características completamente invariantes a partir de mapas de características equivalentes en ambas dimensiones: la espacial (alineación 2D) y la de orientación (conmutación circular de canales e interpolación de características). ReDet logra resultados del estado del arte en DOTA-v1.0 (80.10 mAP), DOTA-v1.5 (76.80 mAP) y HRSC2016 (90.46 mAP under VOC2007) superando ampliamente a los mejores modelos de la literatura con un tamaño de modelo significativamente menor.

## Secciones y Subsecciones

### 1. Introduction
Presenta la motivación del artículo sobre cómo la orientación arbitraria de los objetos aéreos hace ineficientes a las CNN estándar, y propone integrar redes equivalentes a la rotación para reducir la redundancia y mejorar la generalización.
* **Problemas atacados**: La ineficiencia de los detectores de objetos convencionales al procesar objetos en imágenes aéreas con orientaciones arbitrarias, lo que causa alta redundancia en el aprendizaje de características e incrementa el costo de cómputo y el tamaño del modelo.
* **Limitaciones de ese entonces**: Las CNN tradicionales no modelan de forma nativa la variación por rotación (no son equivalentes ante la rotación). Para compensar esto, se suele recurrir a redes muy pesadas y a un incremento artificial del dataset mediante rotación aleatoria, lo que aumenta el tiempo y coste de entrenamiento. Además, los métodos de deformación de RoI rotadas anteriores (RRoI Align o RRoI Pooling) solo logran invariancia en el espacio 2D, dejando desalineados los canales de características.
* **Soluciones alcanzadas**: Se propone *ReDet*, que integra convoluciones de grupo equivalentes a la rotación en el backbone para procesar variaciones angulares de manera predecible, y diseña *RiRoI Align* para lograr invariancia total de características en la fase de propuesta de regiones.

### 2. Related Works
Estudia el desarrollo histórico y conceptual de los tres pilares de este trabajo: detección orientada, redes equivalentes y modelos de invariancia.
* **Problemas atacados**: La falta de simetría por rotación en los detectores aéreos y la desalineación persistente de características en los límites orientados.
* **Limitaciones de ese entonces**: 
  * **2.1. Oriented Object Detection**: El uso de bounding boxes horizontales (HBB) incluye excesivo ruido de fondo. Métodos como RoI Transformer, R3Det o S2A-Net mejoran el alineamiento espacial pero no resuelven la ineficiencia de la convolución estándar frente a rotaciones en el backbone.
  * **2.2. Rotation-equivariant Networks**: Trabajos en redes equivalentes (como convoluciones de grupo sobre grupos discretos $C_4$ o $C_8$) han mejorado la clasificación de imágenes enteras, pero no han sido aplicados con éxito en arquitecturas complejas de detección de objetos.
  * **2.3. Rotation-invariant Object Detection**: Técnicas como STN o DCN proveen invariancia aproximada a nivel de imagen, pero a costa de muchos parámetros adicionales y dependencia excesiva de datos de entrenamiento aumentados.
* **Soluciones alcanzadas**: ReDet combina un backbone equivalente a la rotación con una cabeza de detección equipada con RiRoI Align para proveer invariancia a nivel de instancia completa, optimizando el tamaño y la precisión del modelo.

### 3. Preliminaries
Se establecen las bases matemáticas de la equivalencia e invariancia a la rotación en el procesamiento digital de señales e imágenes.
* **Problemas atacados**: Fundamentación teórica para modelar la invariancia a nivel de instancia en detección de objetos.
* **Limitaciones de ese entonces**: Las CNN estándar solo demuestran equivalencia ante traslaciones, lo que limita su capacidad para transferir el conocimiento aprendido de un patrón en una orientación a otra diferente.
* **Soluciones alcanzadas**: Se define la equivalencia de red $\Phi[T_g(x)] = T_g[\Phi(x)]$ bajo un grupo de transformación $G$ (producto semidirecto de traslaciones y rotaciones discretas $C_N$). Se demuestra formalmente cómo extraer características invariantes ante rotación $\Phi(I_R)$ mediante la transformación inversa $T_r^{-1}$ de la orientación detectada en el mapa de características.

### 4. Rotation-equivariant Detector
Describe el diseño completo de ReDet, integrado por el backbone equivalente, un detector de propuestas horizontales (RPN) y un RoI Transformer para generar propuestas rotadas (RRoIs) que luego procesa RiRoI Align.
* **Problemas atacados**: Pérdida de invariancia de características al usar alineaciones de regiones 2D sobre redes equivalentes a la rotación.
* **Limitaciones de ese entonces**: Al emplear mapas de características con canales de orientación adicionales, las propuestas orientadas (RRoIs) varían en su eje de rotación y desplazan circularmente las respuestas en los canales. Un alineador de región convencional como RRoI Align solo alinea espacialmente, fallando en alinear la dimensión de los canales de orientación.
* **Soluciones alcanzadas**: Se integra un pipeline completo que extrae características equivalentes, propone regiones candidatas poligonales y las alinea espacial y angularmente para alimentar las ramas de clasificación y regresión final.

#### 4.1. Rotation-equivariant Backbone
Detalla el diseño e implementación del extractor de características rotación-equivalente.
* **Problemas atacados**: Redundancia extrema de parámetros en el backbone debido al aprendizaje duplicado de un mismo objeto en diferentes orientaciones.
* **Limitaciones de ese entonces**: Backbones como ResNet o FPN ordinarios carecen de compartición de pesos para orientaciones, requiriendo mayor profundidad y ancho para aproximar robustez angular.
* **Soluciones alcanzadas**: Se implementa *ReResNet con ReFPN* usando la biblioteca `e2cnn`, logrando equivalencia bajo el grupo cíclico $C_N$ (e.g., $C_8$ para 8 ángulos discretos). Los mapas de características resultantes tienen dimensiones $(K, N, H, W)$ con $N$ canales de orientación. Gracias al "rotation weight sharing", los mismos filtros rotados analizan diferentes ángulos, reduciendo a $1/N$ los parámetros del backbone.

#### 4.2. Rotation-invariant RoI Align
Presenta el algoritmo matemático para alinear propuestas orientadas en mapas con canales de orientación.
* **Problemas atacados**: Desalineación de características en la dimensión de orientación al mapear las propuestas.
* **Limitaciones de ese entonces**: Si un objeto rota en la imagen, su respuesta no solo gira espacialmente sino que también se traslada a lo largo de los canales de orientación de la red. Los alineadores tradicionales solo corrigen el giro espacial 2D, dejando las características desalineadas en la dimensión del grupo de rotación.
* **Soluciones alcanzadas**: Se introduce *RiRoI Align*, el cual ejecuta dos operaciones: (1) **Alineación espacial**: un RRoI Align convencional que deforma la región a nivel espacial 2D. (2) **Alineación de orientación**: calcula un desplazamiento de canal $r = \lfloor \theta N / 2\pi \rfloor$ y rota circularmente los canales de orientación (switching channels) para alinear la orientación dominante. Para ángulos continuos $\theta$, realiza una interpolación lineal unidimensional entre los canales vecinos más cercanos.

### 5. Experiments and Analysis
Reporta las pruebas empíricas en DOTA y HRSC2016, evaluando variantes de diseño y comparándolas con la literatura.
* **Problemas atacados**: Demostración empírica de la eficiencia y precisión en la detección de objetos aéreos rotados.
* **Limitaciones de ese entonces**: Falta de claridad en la literatura sobre el impacto de la granularidad de la discretización del grupo cíclico ($C_4, C_8, C_{16}$) y el efecto de la interpolación angular.
* **Soluciones alcanzadas**: Se demuestra que ReDet supera sustancialmente al baseline en precisión y velocidad, con reducciones drásticas del tamaño del modelo.

#### 5.1. Datasets
Detalla los conjuntos de datos de prueba.
* **Problemas atacados**: Medición de robustez ante clases con formas afiladas y escalas muy variables.
* **Limitaciones de ese entonces**: DOTA-v1.5 incluye objetos sumamente pequeños (menores a 10 px) y una clase retadora como Container Crane (CC), lo que dificulta la estabilidad de entrenamiento en detectores convencionales.
* **Soluciones alcanzadas**: DOTA se entrena recortando imágenes en parches de 1024x1024 con solapamiento de 200 píxeles. HRSC2016 se escala a 800x512 preservando la relación de aspecto.

#### 5.2. Implementation Details
Describe el entorno de pruebas, cronogramas de aprendizaje y parámetros óptimos.
* **Problemas atacados**: Convergencia óptima y comparación justa del modelo.
* **Limitaciones de ese entonces**: Entrenar redes con equivalencia explícita en detección es inestable si no se pre-entrenan correctamente en clasificación.
* **Soluciones alcanzadas**: Pre-entrenamiento en ImageNet-1K por 100 épocas. Ajuste fino en detección mediante SGD con learning rate inicial de 0.01 por 12 épocas (DOTA) y 36 épocas (HRSC2016).

#### 5.3. Ablation Studies
Estudios de ablación enfocados en analizar la configuración geométrica y de alineamiento.
* **Problemas atacados**: Determinación del grupo cíclico $C_N$ óptimo y el número de canales de interpolación de RiRoI Align.
* **Limitaciones de ese entonces**: Mayor discretización angular (e.g. $C_{16}$) reduce los parámetros pero puede perjudicar la clasificación al dispersar demasiado las características de orientación.
* **Soluciones alcanzadas**:
  * El grupo $C_8$ proporciona el mejor equilibrio: mejora en 1.83 mAP respecto a la convolución estándar con solo 1/8 de los parámetros del backbone (12 Mb vs 103 Mb).
  * RiRoI Align con interpolación bilineal de $l=2$ canales vecinos obtiene el mAP más alto (66.86%), superando en 0.87% a RRoI Align tradicional. Novedosamente, se muestra que aplicar MaxPooling (Reducción por Máximos) sobre la dimensión de orientación empeora los resultados, confirmando que preservar y alinear todas las orientaciones es beneficioso.
  * Se demuestra que ReDet actúa como un aumento de datos implícito, logrando una mejora de 2.59 mAP frente al baseline con aumento por rotación tradicional y requiriendo un 18% menos de tiempo de entrenamiento.

#### 5.4. Comparisons with the State-of-the-Art
Compara la precisión del modelo contra competidores del estado del arte.
* **Problemas atacados**: Superar a detectores densos o complejos que utilizan arquitecturas masivas.
* **Limitaciones de ese entonces**: Métodos anteriores requieren gran poder de cómputo y backbones gigantescos (ResNet152-FPN o Hourglass-104) para alcanzar alta precisión.
* **Soluciones alcanzadas**: ReDet alcanza 80.10 mAP en DOTA-v1.0 (ganancia de 1.2 mAP), 76.80 mAP en DOTA-v1.5 (ganancia de 3.5 mAP) y 90.46 mAP en HRSC2016, usando un backbone liviano ReResNet50 y recortando un 60% el tamaño total del modelo.

### 6. Conclusions
Resume los logros principales del artículo.
* **Problemas atacados**: Establecer un nuevo estándar para el diseño de detectores de objetos orientados en imágenes aéreas.
* **Limitaciones de ese entonces**: La mayoría de las redes aéreos pasan por alto las propiedades geométricas básicas de la rotación física en imágenes.
* **Soluciones alcanzadas**: Se demostró que integrar simetrías de rotación en la estructura misma de la red (equivarianza en el backbone e invariancia en el alineador de regiones) da como resultado modelos mucho más eficientes, livianos y precisos.
