# Oriented R-CNN for Object Detection

- **Key**: Xie2021OrientedRCNN
- **Year**: 2021
- **Venue**: ICCV 2021 (International Conference on Computer Vision)

## Resumen

Este artículo propone Oriented R-CNN, un framework de detección de objetos orientados en dos etapas que busca superar el cuello de botella computacional presente en los detectores orientados del estado del arte. La motivación principal es que los métodos existentes, como RoI Transformer y Rotated RPN, generan propuestas orientadas mediante procesos costosos en tiempo y parámetros. La solución propuesta consiste en un Oriented RPN (red de propuestas orientada) extremadamente ligero, que genera directamente propuestas orientadas de alta calidad sin prácticamente coste adicional, gracias a una nueva representación llamada "midpoint offset representation" que codifica objetos orientados con seis parámetros. En la segunda etapa, un Oriented R-CNN Head refina y clasifica las propuestas mediante Rotated RoIAlign. Evaluado en DOTA y HRSC2016, el método alcanza 75.87% mAP y 96.50% mAP respectivamente, a 15.1 FPS con ResNet-50-FPN en una sola RTX 2080Ti, superando a todos los detectores del estado del arte en precisión y eficiencia conjuntas.

---

## Secciones y Subsecciones

### 1. Introduction

La introducción contextualiza el problema de la detección de objetos orientados en imágenes aéreas y escenas complejas. Los métodos de detección basados en cajas horizontales (Faster R-CNN) no se adaptan bien a objetos rotados, pues la caja horizontal puede abarcar múltiples objetos o regiones de fondo, provocando inconsistencias entre la confianza de clasificación y la precisión de localización. Hasta entonces, las dos principales estrategias para generar propuestas orientadas eran: (a) Rotated RPN, que coloca 54 anchors rotados por ubicación, causando enorme costo computacional y de memoria; (b) RoI Transformer, que aprende propuestas orientadas desde RoIs horizontales mediante procesos complejos con capas FC y RoI Alignment. La propuesta central del artículo es un Oriented RPN sencillo que genera propuestas de alta calidad cambiando únicamente de 4 a 6 parámetros de salida en la rama de regresión, con solo 1/3000 de los parámetros de RoI Transformer+.

* **Problemas atacados**: Cuello de botella computacional en la generación de propuestas orientadas dentro de los detectores de dos etapas.
* **Limitaciones de ese entonces**: Rotated RPN requería anchors densos y costosos; RoI Transformer introducía capas pesadas de FC y múltiples alineaciones para obtener propuestas orientadas.
* **Soluciones alcanzadas**: Se presenta Oriented R-CNN como framework general eficiente y preciso, con un Oriented RPN de costo casi nulo que supera en velocidad a otros de dos etapas y en precisión a los de una etapa.

---

### 2. Related Work

Esta sección revisa el estado del arte en detección general de objetos y en detección de objetos orientados. Se mencionan detectores de propuestas (Faster R-CNN, R-CNN, Fast R-CNN) y sus variantes orientadas, incluyendo trabajos que emplean anchors rotados (Rotated RPN), representaciones por vértices deslizantes (Gliding Vertex), redes de refinamiento de propuestas (RoI Transformer, SCRDet) y detectores de una etapa o sin anclas (S2ANet, R3Det, DRN, DAL). Se destaca que los detectores de dos etapas que aprenden propuestas orientadas directamente son escasos, complejos y lentos en comparación con los de una etapa.

* **Problemas atacados**: Revisión crítica del gap entre la eficiencia de detectores de una etapa y la precisión de detectores de dos etapas para objetos orientados.
* **Limitaciones de ese entonces**: Los métodos de dos etapas con propuestas orientadas eran pesados (RoI Transformer) o inexactos (detectores con RoIs horizontales), mientras que los de una etapa eran más rápidos pero menos precisos.
* **Soluciones alcanzadas**: El contexto justifica la necesidad de un detector de dos etapas que sea competitivo en velocidad con los de una etapa, pero más preciso.

---

### 3. Oriented R-CNN

Esta sección es el núcleo técnico del artículo y describe en detalle los dos componentes del framework: el Oriented RPN y el Oriented R-CNN Head. El Oriented RPN utiliza la arquitectura FPN estándar con cinco niveles de características y asigna solo 3 anchors horizontales por ubicación espacial, añadiendo dos parámetros extra (∆α, ∆β) en la rama de regresión para capturar la orientación de las propuestas, sin modificar la arquitectura base. El Oriented R-CNN Head toma las propuestas orientadas, extrae características mediante Rotated RoIAlign y las clasifica y regresa con capas FC estándar.

* **Problemas atacados**: Diseñar un oriented RPN que sea a la vez general, simple y de bajo costo computacional, sin sacrificar calidad de propuestas.
* **Limitaciones de ese entonces**: Los métodos previos requerían anchors rotados masivos o redes auxiliares complejas para producir propuestas orientadas.
* **Soluciones alcanzadas**: El Oriented RPN de seis parámetros logra alta calidad de propuestas con una fracción de los parámetros de métodos anteriores.

#### 3.1. Oriented RPN

El Oriented RPN es una red convolucional completamente convolucional y ligera. Toma las cinco escalas de FPN como entrada y predice, por cada ancla, un vector de seis parámetros de regresión: los cuatro clásicos (desplazamiento de centro, escala en ancho y alto) más dos parámetros novedosos (∆α, ∆β) que corresponden a las desviaciones de los vértices superior y derecho de la caja orientada respecto a los puntos medios de los lados de la caja horizontal externa. La clasificación (objectness score) se mantiene como en RPN estándar. Solo se necesitan 3 anchors horizontales por ubicación, lo que reduce enormemente los parámetros respecto a los 54 anchors del Rotated RPN. El entrenamiento sigue el esquema de asignación de muestras positivas/negativas estándar (IoU > 0.7 positivo, < 0.3 negativo), usando la pérdida L1 que combina cross-entropy para clasificación y Smooth L1 para regresión.

* **Problemas atacados**: Generar propuestas orientadas de alta calidad sin el overhead de anchors rotados densos ni módulos auxiliares complejos.
* **Limitaciones de ese entonces**: Rotated RPN usaba 54 anchors rotados; RoI Transformer requería módulos de alineación y capas FC costosas.
* **Soluciones alcanzadas**: El Oriented RPN logra un recall de 92.80% con solo 2000 propuestas, usando 1/3000 de los parámetros de RoI Transformer+.

##### 3.1.1. Midpoint Offset Representation

Se introduce una nueva representación de objetos orientados con seis parámetros: (x, y, w, h, ∆α, ∆β). La caja externa horizontal del objeto orientado se describe por su centro (x, y) y dimensiones (w, h). Los parámetros ∆α y ∆β son los desplazamientos del vértice superior (v1) respecto al punto medio del lado superior, y del vértice derecho (v2) respecto al punto medio del lado derecho, respectivamente. Gracias a la simetría, v3 y v4 se deducen como –∆α y –∆β. Esta representación hereda el mecanismo de regresión horizontal, acota los parámetros de forma natural (los offsets no pueden exceder la mitad del lado), y puede ser decodificada de forma directa para obtener las coordenadas de los cuatro vértices.

* **Problemas atacados**: Representar de forma compacta, acotada y diferenciable cualquier caja orientada arbitraria para su predicción por regresión.
* **Limitaciones de ese entonces**: Las representaciones basadas en ángulo sufrían discontinuidades y problemas de ambigüedad en el borde (angle boundary problem); las representaciones por vértices eran costosas computacionalmente.
* **Soluciones alcanzadas**: La representación de midpoint offset es continua, acotada, no sufre de discontinuidades angulares, y se integra de forma natural en la rama de regresión estándar de RPN.

##### 3.1.2. Loss Function

La función de pérdida del Oriented RPN combina la pérdida de clasificación (cross-entropy) y la pérdida de regresión (Smooth L1) sobre los seis parámetros, siguiendo la convención de Faster R-CNN. La normalización de las transformaciones afines garantiza que los parámetros ∆α y ∆β se escalen de forma coherente con el tamaño del ancla (δα = ∆α/w, δβ = ∆β/h). Se usa un mini-batch de 256 muestras por imagen, con la mitad de muestras positivas como máximo.

* **Problemas atacados**: Supervisar correctamente la predicción de las seis dimensiones de las propuestas orientadas respecto a los anclas horizontales.
* **Limitaciones de ese entonces**: Las pérdidas de regresión angulares anteriores podían ser discontinuas o mal condicionadas en los bordes de ángulo (0°/180°).
* **Soluciones alcanzadas**: La formulación de pérdida mediante transformaciones afines sobre los offsets es estable, continua y compatible con el entrenamiento end-to-end.

#### 3.2. Oriented R-CNN Head

El Oriented R-CNN Head es la segunda etapa del detector. Recibe el mapa de características de FPN (P2–P5) y el conjunto de propuestas orientadas producidas por el Oriented RPN. Para cada propuesta, aplica Rotated RoIAlign para extraer un vector de características de tamaño fijo (7×7×C), que luego pasa por dos capas FC seguidas de dos cabezas: una de clasificación (K+1 clases) y una de regresión de caja orientada. La salida es el conjunto de detecciones finales con sus cajas orientadas y puntuaciones de clase.

* **Problemas atacados**: Refinar las propuestas orientadas y clasificarlas correctamente, eliminando el desalineamiento entre propuesta y características.
* **Limitaciones de ese entonces**: Los detectores previos de dos etapas usaban RoIs horizontales en la segunda etapa, lo que producía desalineamiento de características para objetos rotados.
* **Soluciones alcanzadas**: Rotated RoIAlign extrae características invariantes a la rotación para cada propuesta orientada, mejorando la alineación feature-objeto y la precisión de regresión final.

##### 3.2.1. Rotated RoIAlign

El Rotated RoIAlign adapta la operación RoIAlign estándar para operar sobre regiones orientadas. Dado que las propuestas orientadas del Oriented RPN son paralelogramos, primero se convierten al rectángulo mínimo encubriente con orientación (x, y, w, h, θ). Luego, se proyecta este rectángulo al mapa de características con el stride correspondiente, se divide en una cuadrícula m×m (por defecto 7×7), y para cada celda se aplica bilinear interpolation con transformación de rotación, obteniendo un mapa de características alineado de tamaño fijo.

* **Problemas atacados**: Extraer características alineadas con la orientación del objeto para propuestas arbitrariamente rotadas.
* **Limitaciones de ese entonces**: RoIAlign estándar solo funciona con propuestas horizontales; aplicarlo a objetos rotados producía características con ruido de fondo o de objetos adyacentes.
* **Soluciones alcanzadas**: Rotated RoIAlign extrae características relevantes para el objeto con independencia de su orientación, mejorando la clasificación y regresión final.

#### 3.3. Implementation Details

Los detalles de implementación describen el proceso de entrenamiento end-to-end conjunto del Oriented RPN y del Head, usando SGD con momentum 0.9 y weight decay 1e-4. Se emplean ResNet-50 y ResNet-101 preentrenados en ImageNet. Las imágenes DOTA se recortan en patches de 1024×1024 con stride 824. El NMS horizontal (IoU 0.8) se aplica tras el Oriented RPN para reducir redundancias, seleccionando las top-1000 propuestas para la segunda etapa. El Poly NMS (IoU 0.1) se aplica a las detecciones finales. El entrenamiento usa 12 épocas para DOTA y 36 para HRSC2016.

* **Problemas atacados**: Configurar el sistema para que el pipeline completo sea eficiente en entrenamiento e inferencia.
* **Limitaciones de ese entonces**: El manejo de parches y la fusión de resultados entre parches era necesario dada la alta resolución de las imágenes aéreas.
* **Soluciones alcanzadas**: El pipeline end-to-end con NMS horizontal eficiente logra 15.1 FPS en inferencia, manteniendo alta precisión en las dos etapas.

---

### 4. Experiments

La sección de experimentos evalúa Oriented R-CNN en dos benchmarks estándar: DOTA (detección de objetos orientados en 15 categorías, 2806 imágenes, 188282 instancias) y HRSC2016 (detección de barcos, 1061 imágenes). Se reportan resultados de recall del Oriented RPN (92.80% con 2000 propuestas), comparaciones con 19 y 10 métodos del estado del arte respectivamente, y análisis de velocidad vs. precisión. Oriented R-CNN con R-50-FPN logra 75.87% mAP en DOTA y 96.50% mAP en HRSC2016, superando a todos los competidores incluyendo métodos con backbone R-101-FPN.

* **Problemas atacados**: Demostrar empíricamente que el framework propuesto supera el estado del arte tanto en precisión como en velocidad.
* **Limitaciones de ese entonces**: Los detectores de dos etapas orientados eran más lentos que los de una etapa sin superar su precisión en todos los escenarios.
* **Soluciones alcanzadas**: Oriented R-CNN logra el mejor balance velocidad/precisión: 15.1 FPS con 75.87% mAP, superando a S2ANet (15.3 FPS, 74.12%) y a RoI Transformer+ (11.3 FPS, 74.61%).

#### 4.1. Datasets

Se describen los dos benchmarks: DOTA (15 clases, imágenes de 800×800 a 4000×4000 píxeles, evaluación por servidor oficial) y HRSC2016 (detección de barcos, 300×300 a 1500×900, métricas PASCAL VOC 2007 y 2012).

* **Problemas atacados**: Seleccionar benchmarks representativos y de alta dificultad para validar el método.
* **Limitaciones de ese entonces**: Los benchmarks previos no incluían la diversidad de escalas, densidades y orientaciones presente en imágenes aéreas reales.
* **Soluciones alcanzadas**: Los dos datasets cubren escenarios complementarios (multiclase de alta variedad vs. detección especializada de barcos), dando una evaluación robusta del método.

#### 4.2. Parameter Settings

Se detallan los hiperparámetros de entrenamiento: batch size 2, GPU RTX 2080Ti, backbone ResNet-50/101 (ImageNet pretraining), augmentación por flipping horizontal y vertical, learning rate inicial 0.005 con decaimiento ×0.1 en épocas 8 y 11 (DOTA), y 24 y 33 (HRSC2016). Las imágenes DOTA se recortan con overlap de 200 píxeles para evitar que los objetos queden truncados en los bordes del patch.

* **Problemas atacados**: Reproducibilidad y comparación justa con otros métodos del estado del arte.
* **Limitaciones de ese entonces**: La evaluación sobre imágenes de alta resolución de DOTA requería estrategias de cropping para que los detectores pudieran procesar las imágenes.
* **Soluciones alcanzadas**: El protocolo de evaluación estándar con patching y merging permite comparaciones directas y reproducibles.

#### 4.3. Evaluation of Oriented RPN

Se evalúa el Oriented RPN de forma aislada midiendo el recall sobre el validation set de DOTA. Con 2000, 1000 y 300 propuestas se obtienen 92.80%, 92.20% y 81.60% de recall respectivamente. La pequeña diferencia entre 2000 y 1000 propuestas (0.6%) justifica el uso de 1000 propuestas en inferencia como balance óptimo entre velocidad y recall.

* **Problemas atacados**: Verificar que el Oriented RPN produce propuestas de alta calidad que no limiten el techo de precisión del sistema completo.
* **Limitaciones de ese entonces**: No existían oriented RPNs simples con recall comparable al de métodos complejos como RoI Transformer.
* **Soluciones alcanzadas**: El Oriented RPN logra recall competitivo con muy pocas propuestas, validando su diseño minimalista de seis parámetros.

#### 4.4. Comparison with State-of-the-Arts

La comparación muestra que Oriented R-CNN supera a los 19 métodos comparados en DOTA (incluyendo R3Det, S2ANet, Gliding Vertex, SCRDet, RoI Transformer, etc.) y a los 10 métodos en HRSC2016. Con multi-scale training y testing, alcanza 80.87% mAP en DOTA con R-50-FPN. En HRSC2016, obtiene 96.50% (VOC2012) y 90.40% (VOC2007), siendo el mejor resultado reportado hasta entonces.

* **Problemas atacados**: Demostrar superioridad cuantitativa del método frente al estado del arte.
* **Limitaciones de ese entonces**: Métodos como Gliding Vertex y SCRDet dependían de representaciones o pipelines más complejos para alcanzar precisiones similares.
* **Soluciones alcanzadas**: Oriented R-CNN con backbone más ligero (R-50) supera a métodos con backbone más pesado (R-101), demostrando que la arquitectura propuesta es más eficiente en términos de parámetros vs. precisión.

#### 4.5. Speed versus Accuracy

Se compara el tradeoff velocidad/precisión entre Oriented R-CNN y métodos relevantes (RetinaNet-O, S2ANet, Faster R-CNN-O, RoI Transformer+), todos con R-50-FPN en la misma GPU. Oriented R-CNN es el más preciso (75.87% mAP) y el más rápido de los de dos etapas (15.1 FPS), siendo comparable en velocidad con los de una etapa pero significativamente más preciso.

* **Problemas atacados**: Mostrar que la mejora en precisión no se obtiene a costa de la eficiencia computacional.
* **Limitaciones de ese entonces**: Los detectores de dos etapas orientados eran más precisos pero mucho más lentos, lo que limitaba su uso en aplicaciones de tiempo real.
* **Soluciones alcanzadas**: El diseño minimalista del Oriented RPN permite que el sistema de dos etapas compita en velocidad con los de una etapa, eliminando el cuello de botella de la generación de propuestas.

---

### 5. Conclusions

El artículo concluye que Oriented R-CNN es un detector práctico de dos etapas para objetos orientados arbitrarios, que combina alta precisión y velocidad competitiva. Los experimentos demuestran su superioridad frente al estado del arte en dos benchmarks representativos. Los autores esperan que el trabajo inspire nuevos diseños de detectores orientados y sirva como baseline sólido para la comunidad. El código está disponible en GitHub (OBBDetection).

* **Problemas atacados**: Sintetizar las contribuciones del trabajo y su impacto esperado en la comunidad.
* **Limitaciones de ese entonces**: El campo carecía de un baseline simple, rápido y de alta precisión para detección orientada de dos etapas.
* **Soluciones alcanzadas**: Oriented R-CNN llena este vacío y proporciona un punto de referencia accesible para futuros trabajos en detección de objetos orientados.
