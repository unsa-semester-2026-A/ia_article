# ReDet: A Rotation-equivariant Detector for Aerial Object Detection

- **Key**: Han2021ReDet
- **Year**: 2021
- **Venue**: CVPR 2021 (IEEE/CVF Conference on Computer Vision and Pattern Recognition)

## Resumen

Este artículo propone ReDet (Rotation-equivariant Detector), un detector de objetos para imágenes aéreas que incorpora explícitamente equivarianza a la rotación e invarianza a la rotación en toda la arquitectura. La motivación central es que las CNN convencionales no modelan explícitamente la variación de orientación, por lo que los detectores orientados requieren una gran cantidad de parámetros y datos augmentados para aprender a manejar objetos con orientaciones arbitrarias. ReDet resuelve esto mediante dos componentes: (1) un backbone de redes rotation-equivariant (ReResNet+ReFPN) que produce mapas de características con múltiples canales de orientación, y (2) Rotation-invariant RoI Align (RiRoI Align), que extrae características completamente invariantes a la rotación en dimensiones espacial y de orientación a partir de los mapas equivariantes. Los experimentos en DOTA-v1.0, DOTA-v1.5 y HRSC2016 muestran que ReDet supera el estado del arte con 80.10, 76.80 y 90.46 mAP respectivamente, mientras reduce el número de parámetros del backbone en un 60% (de 313 MB a 121 MB).

---

## Secciones y Subsecciones

### 1. Introduction

La introducción establece el problema de la detección de objetos en imágenes aéreas, donde los objetos están distribuidos con orientaciones arbitrarias. Las CNN convencionales no son equivariantes a la rotación, es decir, alimentar una imagen rotada produce mapas de características distintos a rotar los mapas de la imagen original. Esto implica que los detectores orientados necesitan muchos parámetros redundantes para modelar la variación de orientación, y grandes cantidades de datos aumentados con rotación para aprender representaciones robustas. Los métodos previos como RRoI Pooling y RRoI Align intentan extraer características invariantes a la rotación mediante warping espacial, pero no pueden lograr invarianza completa pues los canales de orientación quedan desalineados. El artículo propone encadenar redes equivariantes a la rotación con RiRoI Align para resolver ambos problemas de forma integral.

* **Problemas atacados**: Modelado ineficiente de la variación de orientación en detectores aéreos basados en CNN, que requiere parámetros redundantes y datos aumentados costosos.
* **Limitaciones de ese entonces**: Las CNN convencionales no son rotation-equivariant; los RRoI Align existentes solo alinean en la dimensión espacial, dejando los canales de orientación desalineados.
* **Soluciones alcanzadas**: ReDet integra redes rotation-equivariant en el backbone y RiRoI Align para extracción completamente rotation-invariant, logrando el estado del arte con menor número de parámetros.

---

### 2. Related Works

#### 2.1. Oriented Object Detection

Se revisan los detectores orientados existentes: algunos adoptan anchors rotados con múltiples ángulos (aumentando la complejidad), otros como RoI Transformer transforman HRoIs en RRoIs, Gliding Vertex y CenterMap usan cuadriláteros y mapas de probabilidad, R3Det y S2ANet alinean features entre campos receptivos horizontales y anchors rotados, CSL trata la predicción angular como clasificación, y métodos basados en CenterNet (DRN, BBAVectors) exploran la detección anchor-free. La distinción clave de ReDet es que actúa a nivel de toda la red, desde el backbone hasta el detection head, no solo en la etapa de refinamiento.

* **Problemas atacados**: Mejorar la representación de objetos orientados a lo largo de toda la red, no solo en etapas de refinamiento.
* **Limitaciones de ese entonces**: Los métodos previos mejoraban la representación del objeto (ángulo, vértice, etc.) pero no abordaban la equivarianza intrínseca de la red.
* **Soluciones alcanzadas**: ReDet produce features equivariantes en el backbone para que la información de orientación se transmita naturalmente a lo largo de toda la red.

#### 2.2. Rotation-equivariant Networks

Se describe el marco teórico de las redes equivariantes a la rotación, comenzando con las group convolutions de Cohen y Welling, que extienden las CNN al grupo cíclico CN de rotaciones discretas. HexaConv extiende esto a retículas hexagonales. Otros métodos usan filtros armónicos o re-muestreo por interpolación para lograr equivarianza en el dominio continuo. Estos métodos han tenido éxito en clasificación, pero ReDet es el primero en aplicarlos sistemáticamente a detección de objetos orientados.

* **Problemas atacados**: Extender la equivarianza a la rotación, bien establecida en clasificación, al dominio de la detección de objetos orientados.
* **Limitaciones de ese entonces**: Las redes rotation-equivariant se habían aplicado principalmente a clasificación y no a detección, donde la localización de instancias individuales requiere tratamiento diferente.
* **Soluciones alcanzadas**: ReDet adopta e2cnn para implementar el backbone equivariante y extiende el concepto al pipeline de detección completo.

#### 2.3. Rotation-invariant Object Detection

Se discuten los métodos previos que buscan extracción de características invariantes a la rotación a nivel de instancia para detección: STN, DCN, RRoI Pooling, RRoI Align, etc. Se argumenta que el warping ordinario con CNN convencionales produce características aproximadamente invariantes, pero no exactamente, ya que los mapas de características de las CNN no son equivariantes.

* **Problemas atacados**: Obtener características verdaderamente invariantes a la rotación a nivel de instancia para objetos de orientación arbitraria.
* **Limitaciones de ese entonces**: El RRoI Align convencional solo alinea la dimensión espacial; los canales de orientación permanecen desalineados al usar CNN convencionales como backbone.
* **Soluciones alcanzadas**: RiRoI Align resuelve ambas dimensiones (espacial y de orientación) gracias a los mapas equivariantes del backbone.

---

### 3. Preliminaries

Esta sección establece el marco matemático formal de la equivarianza e invarianza a la rotación. Se define la equivarianza como la propiedad por la cual una transformación aplicada a la entrada produce una transformación predecible de la salida: Φ[TgX(x)] = TgY[Φ(x)]. Se recuerda que las CNN son equivariantes a la traslación y se explica cómo las group convolutions extienden esta propiedad al grupo cíclico CN de rotaciones discretas. Se formaliza la equivarianza de las redes rotation-equivariant completas: rotar la entrada es equivalente a rotar los mapas de características en todos los niveles de la red. Finalmente, se deriva cómo esta propiedad permite obtener características invariantes a la rotación a nivel de instancia aplicando la transformación inversa Tr' al feature de la región rotada.

* **Problemas atacados**: Establecer el fundamento teórico que justifica el diseño de RiRoI Align y el backbone equivariante.
* **Limitaciones de ese entonces**: Los métodos de detección no habían formalizado la relación entre equivarianza de la red y la extracción de características invariantes por instancia.
* **Soluciones alcanzadas**: La formulación matemática demuestra que si el backbone es equivariante, se puede obtener la característica de cualquier RoI como si estuviera sin rotar, aplicando solo la transformación inversa.

---

### 4. Rotation-equivariant Detector

Esta sección describe la arquitectura completa de ReDet. El flujo general es: imagen → backbone equivariante (ReResNet+ReFPN) → RPN+RoI Transformer → RiRoI Align → clasificación y regresión de bbox. El backbone produce mapas de características con dimensión adicional de orientación (K, N, H, W), con N canales de orientación. Estos mapas equivariantes permiten que RiRoI Align extraiga características completamente invariantes alineando tanto la dimensión espacial como la de orientación.

* **Problemas atacados**: Integrar redes rotation-equivariant con el pipeline de detección de dos etapas de forma eficiente.
* **Limitaciones de ese entonces**: No existía un detector de objetos orientados que incorporase equivarianza a la rotación de forma sistemática en todas las capas.
* **Soluciones alcanzadas**: ReDet unifica el backbone equivariante y RiRoI Align en un pipeline end-to-end que logra equivarianza en extracción de features e invarianza en la cabeza de detección.

#### 4.1. Rotation-equivariant Backbone

El backbone equivariante (ReResNet) reimplementa todas las capas de ResNet (convolución, pooling, normalización, no-linealidades) usando redes equivariantes basadas en e2cnn. El backbone es equivariante al grupo (R², +) ⋊ CN (todas las traslaciones y N rotaciones discretas). Los mapas de características tienen N canales de orientación, donde cada canal corresponde a un elemento del grupo CN. Las ventajas sobre el backbone convencional son: (a) mayor compartición de pesos (incluyendo rotación), lo que reduce drásticamente el tamaño del modelo (≈1/N parámetros); (b) información de orientación enriquecida: para una imagen con orientación fija, se producen features desde múltiples orientaciones; (c) modelo más pequeño: con C8 (8 rotaciones discretas), ReResNet50+ReFPN reduce el backbone de 103 MB a 12 MB manteniendo mejoras en detección.

* **Problemas atacados**: Diseñar un backbone que produzca features equivariantes y sea eficiente en parámetros.
* **Limitaciones de ese entonces**: Los backbones ResNet convencionales no son equivariantes a la rotación y requieren grandes capacidades de red para modelar implícitamente las variaciones de orientación.
* **Soluciones alcanzadas**: ReResNet50+ReFPN con C8 logra 1.83 mAP más que ResNet50+FPN usando solo 1/8 de los parámetros del backbone, demostrando alta eficiencia de parámetros.

#### 4.2. Rotation-invariant RoI Align

RiRoI Align es el componente central para extraer características verdaderamente invariantes a la rotación desde los mapas equivariantes. Consiste en dos etapas: (a) Alineación espacial: igual al RRoI Align convencional, warpea el RRoI en la dimensión espacial; (b) Alineación de orientación: dado el ángulo θ del RRoI, calcula el índice r = ⌊θN/2π⌋ para cambiar cíclicamente los canales de orientación (switching channels, SC), de modo que el canal C_N^(r) pase a ser el primer canal. Adicionalmente, si θ no es múltiplo exacto de 2π/N, se interpola entre los canales de orientación más cercanos con factor α = θN/2π - r. La comparación con RRoI Align+MaxPool (orientación pooling) demuestra que preservar todos los canales de orientación y alinearlos es superior a colapsar la dimensión de orientación.

* **Problemas atacados**: Extraer características invariantes a la rotación tanto en la dimensión espacial como en la de orientación, aprovechando los mapas equivariantes del backbone.
* **Limitaciones de ese entonces**: RRoI Align convencional solo alinea la dimensión espacial; el orientation pooling (MaxPool sobre canales de orientación) pierde información crucial de orientaciones débiles.
* **Soluciones alcanzadas**: RiRoI Align con l=2 interpolaciones logra +0.87 mAP sobre RRoI Align estándar y +1.39 mAP sobre RRoI Align+MaxPool, validando la importancia de preservar y alinear todos los canales de orientación.

---

### 5. Experiments and Analysis

#### 5.1. Datasets

Se utilizan tres benchmarks: DOTA-v1.0 (2806 imágenes aéreas, 188282 instancias, 15 categorías), DOTA-v1.5 (402089 instancias, incluye la categoría CC y más instancias extremadamente pequeñas, más desafiante) y HRSC2016 (1061 imágenes de barcos de alta resolución). Se aplica la estrategia estándar de cropping en patches de 1024×1024 con stride 824, augmentación con flipping horizontal y datos multi-escala (0.5, 1.0, 1.5) para comparaciones equitativas.

* **Problemas atacados**: Validar el método en benchmarks con diferentes niveles de dificultad (escala, densidad de objetos pequeños, orientaciones arbitrarias).
* **Limitaciones de ese entonces**: DOTA-v1.0 era el estándar pero DOTA-v1.5 planteaba dificultades adicionales con más instancias pequeñas (menos de 10 píxeles).
* **Soluciones alcanzadas**: ReDet demuestra generalización en todos los benchmarks, con mejoras especialmente en categorías de instancias pequeñas y de alta variación de escala.

#### 5.2. Implementation Details

ReDet se implementa sobre mmdetection. El backbone ReResNet se preentrenó en ImageNet-1K con 100 épocas. Para el fine-tuning de detección, se usa SGD con lr inicial 0.01 (dividida por 10 en cada decaimiento), momentum 0.9 y weight decay 1e-4, 12 épocas para DOTA y 36 para HRSC2016. Se configuran 15 anchors por nivel FPN, 512 RoIs por imagen (ratio positivo:negativo 1:3). La evaluación utiliza 10000 RoIs antes del NMS y 2000 después. El entrenamiento emplea 4 GPUs V100 con batch size total 8.

* **Problemas atacados**: Establecer un protocolo de entrenamiento reproducible y comparable con el estado del arte.
* **Limitaciones de ese entonces**: El preentrenamiento en ImageNet para redes equivariantes no estaba disponible de forma directa, requiriendo entrenamiento desde cero de ReResNet.
* **Soluciones alcanzadas**: El protocolo de preentrenamiento propio de ReResNet en ImageNet permite fine-tuning eficiente para detección, siendo comparable con los métodos basados en ResNet estándar.

#### 5.3. Ablation Studies

Se realizan ablaciones sobre DOTA-v1.5 para evaluar: (a) Backbone equivariante: C8 ofrece el mejor balance precisión-parámetros (1.83 mAP más que ResNet50 con 1/8 de parámetros); C16 pierde precisión de clasificación excesiva. (b) RiRoI Align: con l=2 interpolaciones logra el mejor resultado (+0.87 vs. RRoI Align), mientras que MaxPool de orientación es perjudicial (−1.39 mAP). (c) Comparación con rotación augmentada: ReDet (con parámetros similares al baseline, ReDet*) supera al baseline con rotación en +2.59 mAP con solo 18% más tiempo de entrenamiento.

* **Problemas atacados**: Cuantificar la contribución individual de cada componente (backbone equivariante y RiRoI Align) y comparar con estrategias alternativas de aumento de datos.
* **Limitaciones de ese entonces**: No se sabía si la equivarianza de red podría reemplazar eficientemente la augmentación con rotaciones.
* **Soluciones alcanzadas**: La ablación demuestra que ambos componentes son complementarios y esenciales, y que ReDet ofrece mejor balance precisión/parámetros que la augmentación con rotación.

#### 5.4. Comparisons with the State-of-the-Art

ReDet supera todos los métodos previos en los tres benchmarks: en DOTA-v1.0 (single-scale: 76.25 mAP, multi-scale: 80.10 mAP), en DOTA-v1.5 (single-scale: 66.86 OBB mAP, 67.66 HBB mAP; multi-scale: 76.80 OBB mAP), y en HRSC2016 (90.46/97.63 mAP VOC2007/VOC2012). En DOTA-v1.0, el single-scale de ReDet supera incluso a modelos multi-scale previos. En DOTA-v1.5, las mejoras son más pronunciadas en categorías difíciles (HA, SP, CC) con alta variación de escala.

* **Problemas atacados**: Demostrar la superioridad empírica de ReDet frente al estado del arte en múltiples benchmarks de detección aérea.
* **Limitaciones de ese entonces**: Los mejores métodos previos (S2ANet, CSL, SCRDet++) dependían de CNN estándar con técnicas adicionales de alineación de features, pero no abordaban la equivarianza fundamental de la red.
* **Soluciones alcanzadas**: ReDet establece nuevo estado del arte en los tres benchmarks con una reducción de parámetros del 60%, demostrando que la equivarianza a la rotación es una propiedad fundamental y eficiente para la detección aérea.

---

### 6. Conclusions

El artículo concluye que ReDet, al incorporar explícitamente equivarianza e invarianza a la rotación mediante el backbone ReResNet+ReFPN y RiRoI Align, logra superar el estado del arte en detección de objetos aéreos con mayor eficiencia de parámetros. Los experimentos extensivos en DOTA y HRSC2016 confirman la efectividad y generalización del método. El código está disponible en GitHub para facilitar la reproducibilidad.

* **Problemas atacados**: Sintetizar las contribuciones del trabajo y su relevancia para el campo de la detección aérea.
* **Limitaciones de ese entonces**: Los detectores de objetos orientados previos no aprovechaban las propiedades matemáticas de equivarianza a la rotación para diseñar redes más eficientes.
* **Soluciones alcanzadas**: ReDet demuestra que incorporar equivarianza a la rotación de forma sistemática es posible, beneficioso para la precisión y altamente eficiente en parámetros, estableciendo un nuevo paradigma para el diseño de detectores orientados.
