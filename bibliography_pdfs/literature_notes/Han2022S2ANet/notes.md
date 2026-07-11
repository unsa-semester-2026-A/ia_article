# Align Deep Features for Oriented Object Detection

- **Key**: Han2022S2ANet
- **Year**: 2022
- **Venue**: IEEE Transactions on Geoscience and Remote Sensing (TGRS) — DOI: 10.1109/TGRS.2021.3062048

---

## Resumen

Este artículo presenta **S²A-Net** (Single-Shot Alignment Network), una red de detección de objetos orientados en imágenes aéreas que ataca de manera explícita el problema de *misalignment* (desalineación) entre las características convolucionales fijas y los objetos con orientaciones arbitrarias. La motivación central es que los detectores de una sola etapa (one-stage) sufren dos tipos de desalineación: (1) los anclajes heurísticos cubren mal los objetos orientados y de alta relación de aspecto, causando desequilibrio entre clases; y (2) las convoluciones estándar tienen campos receptivos alineados con los ejes, mientras que los objetos en imágenes aéreas aparecen a orientaciones arbitrarias, lo que produce inconsistencia entre la puntuación de clasificación y la precisión de localización. Para resolver esto, S²A-Net combina un **Módulo de Alineación de Características (FAM)** —que refina los anclajes horizontales en rotados con una *Anchor Refinement Network* (ARN) y extrae características alineadas con una novedosa *Alignment Convolution* (AlignConv)— y un **Módulo de Detección Orientada (ODM)** —que usa *Active Rotating Filters* (ARF) para generar características sensibles a la orientación y luego las comprime en características invariantes a la orientación. Los experimentos en DOTA y HRSC2016 muestran que S²A-Net alcanza 79.42% mAP y 95.01% mAP respectivamente, superando al estado del arte en velocidad y exactitud simultáneamente.

---

## Secciones y Subsecciones

### I. Introduction

Esta sección contextualiza el problema de detección de objetos en imágenes aéreas (ODAI), explicando por qué los métodos existentes basados en R-CNN de dos etapas y los detectores de una etapa fallan cuando los objetos presentan grandes variaciones de escala, orientaciones arbitrarias y distribuciones densas. Se argumenta que el mayor obstáculo no es simplemente la elección de anclajes, sino la desalineación sistemática entre los anclajes y las características convolucionales extraídas por la red. Posteriormente se presentan las contribuciones principales del paper: la AlignConv, el S²A-Net completo, y el resultado de 79.42% mAP en DOTA.

* **Problemas atacados**: Desalineación entre anclajes heurísticos y objetos orientados; inconsistencia entre puntuación de clasificación y precisión de localización en detectores de una etapa aplicados a imágenes aéreas.
* **Limitaciones de ese entonces**: Los detectores de dos etapas (Faster R-CNN, RoI Transformer) requerían anclajes orientados en gran número y operaciones de RoI complejas, sacrificando velocidad. Los de una etapa (RetinaNet, YOLO) eran rápidos pero sufrían grave desalineación debido a que los anclajes horizontales no podían representar bien objetos orientados y de alta relación de aspecto, y sus características convolucionales fijas no se adaptaban a las orientaciones arbitrarias.
* **Soluciones alcanzadas**: Se propone un detector de una etapa con doble módulo (FAM + ODM) que logra alineación completa de características y reduce la inconsistencia clasificación–localización, alcanzando estado del arte en velocidad y exactitud sin necesidad de grandes conjuntos de anclajes.

---

### II. Related Works

#### A. Object Detection in Aerial Images

Esta subsección revisa los métodos previos para detección en imágenes aéreas, divididos según cómo manejan las orientaciones: métodos con anclajes horizontales que sufren misalignment grave cuando hay objetos densos y orientados; métodos con anclajes rotados que reducen el problema pero aumentan el coste computacional; y métodos híbridos como RoI Transformer que transforma RoIs horizontales en rotados evitando muchos anclajes pero aún necesitando operaciones RoI complejas. También se menciona a Gliding Vertex (Xu et al.) que desliza vértices del bounding box horizontal, pero cuya característica de RoI sigue siendo horizontal y padece misalignment. Finalmente se describe R3Det que muestrea características desde cinco ubicaciones del anclaje para recodificar información de posición.

* **Problemas atacados**: ¿Cómo representar y detectar objetos con orientaciones arbitrarias y gran variación de escala en imágenes aéreas sin incurrir en costes de anclaje extremos?
* **Limitaciones de ese entonces**: Los métodos con anclajes rotados requerían decenas de ángulos, escalas y relaciones de aspecto, disparando el coste computacional. Los métodos basados en RoI aún dependían de operaciones de warping/interpolación costosas. Ningún método lograba alineación completa en una sola etapa.
* **Soluciones alcanzadas**: El survey establece que la brecha de rendimiento entre detectores de una y dos etapas se debe principalmente a la falta de alineación de características, motivando el enfoque de S²A-Net.

#### B. Feature Alignment in Object Detection

Revisa cómo se ha abordado la alineación de características en detectores generales: RoIPooling cuantiza límites flotantes causando misalignment; RoIAlign usa interpolación bilineal para mayor precisión pero sigue siendo costoso; Deformable RoIPooling añade offsets a las subregiones. En detectores de una etapa, Guided Anchoring aprende un campo de offsets para guiar DeformConv, y AlignDet diseña una RoI Convolution que emula RoIAlign. Se critica que todos estos métodos funcionan bien en imágenes naturales pero fallan con objetos densos y orientados en imágenes aéreas porque el campo de offsets se aprende con supervisión débil o no está condicionado a la orientación del anclaje.

* **Problemas atacados**: Cómo lograr alineación de características en detectores de una etapa sin el overhead de las operaciones RoI de dos etapas, y cómo hacerlo de manera condicionada a la orientación del objeto.
* **Limitaciones de ese entonces**: DeformConv aprende offsets implícitamente con supervisión débil, lo que puede fallar para objetos densamente empaquetados y orientados. Guided Anchoring y AlignDet funcionan bien para objetos en imágenes naturales pero no aprovechan la información geométrica del anclaje orientado.
* **Soluciones alcanzadas**: La AlignConv propuesta infiere el campo de offsets directamente desde la caja del anclaje (de forma determinista), garantizando que los puntos de muestreo estén dentro de la región del objeto orientado.

#### C. Inconsistency between Regression and Classification

Explica el problema fundamental de que el score de clasificación de un detector no siempre refleja la precisión de localización: detecciones con score alto pueden tener bounding boxes imprecisos, mientras que otros con alta precisión de localización son suprimidos por NMS. Se revisan IoU-Net (aprende a predecir el IoU de cada detección) y DoubleHead R-CNN (usa cabezas distintas para clasificación y regresión). En S²A-Net, se aborda este problema extrayendo características alineadas por instancia (mejora el score de clasificación) y usando ARF para generar características sensibles y luego invariantes a la orientación, asignando la característica correcta a cada tarea.

* **Problemas atacados**: La inconsistencia entre score de clasificación y precisión de localización en la fase de post-procesamiento (NMS).
* **Limitaciones de ese entonces**: IoU-Net y DoubleHead R-CNN exigen cabezas adicionales o predicciones de IoU complejas que aumentan el coste. Las características compartidas del backbone son subóptimas para ambas tareas simultáneamente.
* **Soluciones alcanzadas**: S²A-Net usa ARF para producir características sensibles a la orientación para regresión e invariantes para clasificación, mitigando la inconsistencia sin añadir un predictor de IoU explícito.

---

### III. Proposed Method

Esta sección describe en detalle la arquitectura completa de S²A-Net, comenzando por la conversión de RetinaNet en un detector orientado como línea base, luego la AlignConv, el FAM, el ODM, y finalmente el ensamblaje completo (entrenamiento e inferencia). La arquitectura general usa un backbone + FPN como extractor de características piramidales, sobre el que se aplica el cabezal de detección (FAM + ODM) en cada escala.

#### A. RetinaNet as Baseline

Se toma RetinaNet, un detector representativo de una etapa, como baseline. Se mantiene su backbone (FPN + ResNet), sus subredes de clasificación y regresión, y su Focal Loss. La única modificación es reemplazar la salida de bounding box horizontal (x, w, h) por una orientada (x, w, h, θ), donde θ ∈ [−π/4, 3π/4] denota el ángulo desde la dirección de x₁ hacia la dirección del ancho w. Todos los demás hiperparámetros se mantienen.

* **Problemas atacados**: Adaptar un detector de una etapa genérico al problema de detección orientada con mínima modificación estructural.
* **Limitaciones de ese entonces**: RetinaNet usaba 9 anclajes por ubicación (3 escalas × 3 relaciones de aspecto) y bounding boxes horizontales, lo que causaba misalignment con objetos orientados.
* **Soluciones alcanzadas**: Con solo cambiar el formato de salida a bounding box orientado y usar un anclaje cuadrado por ubicación, se establece una línea base sólida que evita el overhead de múltiples anclajes rotados.

#### B. Alignment Convolution

La AlignConv es la contribución técnica central del artículo. Se parte de la convolución estándar 2D (que muestrea con una grilla regular R) y de la convolución deformable (que aprende un campo de offsets O). La AlignConv también añade un campo de offsets, pero en lugar de aprenderlo, lo **calcula deterministicamente** a partir del anclaje orientado (x, w, h, θ) correspondiente a cada ubicación p. Para cada punto r ∈ R de la grilla 3×3, la ubicación de muestreo basada en el anclaje es:

```
L^r_p = x + (1/S)(w, h) · r · R^T(θ)
```

donde S es el stride del mapa de características y R(θ) es la matriz de rotación. El campo de offsets O se calcula como la diferencia entre las ubicaciones basadas en el anclaje y las regulares. Así, los 9 puntos de muestreo de la convolución se reorientan para quedar dentro de la región del objeto orientado. La AlignConv introduce un overhead negligible respecto a la convolución estándar porque el cálculo del offset es determinista, a diferencia del DeformConv que requiere una red adicional para aprender el offset con supervisión débil.

* **Problemas atacados**: Desalineación entre las características convolucionales (fijas y alineadas con los ejes) y los objetos con orientación arbitraria; muestreo de características fuera de la región del objeto en objetos densos.
* **Limitaciones de ese entonces**: La convolución estándar muestrea con grilla regular, ignorando la orientación. La convolución deformable aprende offsets implícitamente con supervisión débil, lo que puede llevar a muestreo incorrecto en escenarios de objetos densos.
* **Soluciones alcanzadas**: La AlignConv garantiza que los puntos de muestreo estén dentro del bounding box orientado del anclaje, logrando alineación precisa con coste computacional mínimo (~1.41 GFLOPs adicionales vs. ~3% de mejora en mAP).

#### C. Feature Alignment Module (FAM)

El FAM combina la ARN con una capa de AlignConv (ACL). La **Anchor Refinement Network (ARN)** es una red ligera con dos ramas paralelas: una rama de clasificación de anclajes y una rama de regresión. La rama de regresión transforma los anclajes cuadrados horizontales (θ=0) en anclajes rotados de alta calidad (x, w, h, θ). En la fase de inferencia, la rama de clasificación se descarta para mayor velocidad. Se usa un solo anclaje cuadrado por ubicación (en contraste con los 9 de RetinaNet estándar), y las predicciones de baja confianza no se filtran porque algunas se vuelven positivas en ODM. La **Alignment Convolution Layer (ACL)** toma el mapa de predicción de anclajes (H×W×5) y la característica de entrada, calcula el campo de offsets de 18 dimensiones (9 puntos × 2 offsets), y produce características alineadas mediante AlignConv.

* **Problemas atacados**: Generación de anclajes de alta calidad que representen bien la orientación y forma de los objetos, y extracción de características que correspondan geométricamente al anclaje orientado.
* **Limitaciones de ese entonces**: Los detectores previos usaban decenas de anclajes orientados predefinidos con alto coste, o dependían de operaciones RoI complejas para alinear características.
* **Soluciones alcanzadas**: El FAM genera anclajes de alta calidad con una red ligera y alinea las características en un paso convolucional totalmente convolucional, sin operaciones de warping o RoI, manteniendo alta eficiencia.

#### D. Oriented Detection Module (ODM)

El ODM aborda la inconsistencia entre score de clasificación y precisión de localización. Primero aplica **Active Rotating Filters (ARF)** —filtros k×k×N que rotan N−1 veces durante la convolución para producir un mapa de características con N canales de orientación— para codificar la información de orientación y obtener **características sensibles a la orientación** (orientation-sensitive features). Estas características se alimentan a la subred de regresión. Luego, para obtener **características invariantes a la orientación** (orientation-invariant features) para clasificación, se hace max-pooling sobre los N canales de orientación, seleccionando el canal con mayor respuesta. Este pooling reduce el tamaño del mapa de H×W×256 (con 8 canales de orientación) a H×W×32, siendo eficiente. Finalmente, las características invariantes se alimentan a la subred de clasificación.

* **Problemas atacados**: Inconsistencia entre la puntuación de clasificación y la precisión de localización; las redes CNN no son invariantes a la rotación por naturaleza, lo que perjudica la clasificación de objetos orientados.
* **Limitaciones de ese entonces**: Los métodos previos usaban las mismas características para regresión y clasificación. IoU-Net y DoubleHead R-CNN requerían módulos adicionales complejos para resolver esta inconsistencia.
* **Soluciones alcanzadas**: El ODM resuelve la inconsistencia de forma elegante: usa características sensibles a la orientación para la regresión (que se beneficia de saber la dirección del objeto) e invariantes a la orientación para la clasificación (que necesita ser robusta a variaciones de pose).

#### E. Single-Shot Alignment Network

Esta subsección ensambla el sistema completo describiendo el entrenamiento e inferencia de S²A-Net. Los **targets de regresión** se parametrizan de forma estándar con fórmulas logarítmicas para escalas y diferencias angulares. La **estrategia de matching** usa IoU entre bounding boxes orientados (en lugar de horizontales), con umbral de foreground=0.5 y background=0.4. La **función de pérdida** combina la pérdida del FAM y del ODM: para cada componente, se usa Focal Loss para clasificación y Smooth L1 para regresión, balanceadas por el parámetro λ=1. En **inferencia**, la imagen se pasa por el backbone para extraer características piramidales, el FAM produce anclajes refinados y características alineadas, el ODM produce predicciones de alta confianza, y finalmente se seleccionan los top-2000 y se aplica NMS para obtener las detecciones finales.

* **Problemas atacados**: Cómo integrar coherentemente el FAM y el ODM en un detector de una sola etapa, definir los objetivos de entrenamiento para bounding boxes orientados, y garantizar eficiencia en inferencia.
* **Limitaciones de ese entonces**: Los detectores orientados previos usaban matching de IoU horizontal o requerían conversiones complejas a coordenadas orientadas. La inferencia en imágenes grandes (4000×4000) suponía cuellos de botella de memoria.
* **Soluciones alcanzadas**: S²A-Net es totalmente convolucional y puede inferir sobre imágenes grandes directamente, sin operaciones de RoI. La red es 22.6 FPS en ResNet50 y alcanza 74.12% mAP en DOTA sin augmentación.

---

### IV. Experiments and Analysis

#### A. Datasets

Se describen los dos conjuntos de datos utilizados:

- **DOTA**: Dataset de imágenes aéreas a gran escala con 2806 imágenes (de 800×800 a 4000×4000), 188,282 instancias en 15 categorías de objetos orientados. Para el entrenamiento se recortan parches de 1024×1024 con stride de 824. Se usan conjuntos de entrenamiento y validación para entrenar, y el de prueba para evaluar.
- **HRSC2016**: Dataset de reconocimiento de barcos en alta resolución con 1061 imágenes (de 300×300 a 1500×900), anotadas con bounding boxes orientados. Se redimensionan a (800,512) y se usan 436+181 imágenes para entrenar y 444 para probar.

* **Problemas atacados**: Evaluación en condiciones realistas con objetos de alta variabilidad de tamaño, densidad y orientación.
* **Limitaciones de ese entonces**: DOTA era el benchmark de referencia pero no había un estándar para evaluar en imágenes de tamaño original (sin recorte en chips), lo que sesgaba las comparaciones hacia el procesamiento por parches.
* **Soluciones alcanzadas**: Se establece un protocolo de evaluación claro incluyendo experimentos con y sin data augmentation, y con distintas estrategias de recorte.

#### B. Implementation Details

Se detallan los hiperparámetros de entrenamiento: backbone ResNet101 FPN (para comparación justa) y ResNet50 FPN (para experimentos propios); un anclaje cuadrado por ubicación; Focal Loss con α=0.25 y γ=2.0; 12 épocas para DOTA y 36 para HRSC2016; SGD con lr inicial=0.01, momentum=0.9 y weight decay=0.0001; warmup de 500 iteraciones; 4 GPUs V100 con batch size total=8.

* **Problemas atacados**: Garantizar reproducibilidad y comparación justa con métodos del estado del arte.
* **Limitaciones de ese entonces**: Muchos trabajos usaban configuraciones distintas (número de anclajes, backbones, augmentación) dificultando comparaciones directas.
* **Soluciones alcanzadas**: Se adopta el framework mmdetection como base y se estandarizan configuraciones, permitiendo comparaciones directas.

#### C. Ablation Studies

Esta subsección valida mediante experimentos controlados la contribución de cada componente:

- **RetinaNet como baseline**: La versión ligera (1 anclaje, 2 capas de cabezal) alcanza 67.00% mAP con 156.33 GFLOPs y 33.69 M parámetros, mostrando que el baseline reducido sigue siendo competitivo.
- **Efectividad de AlignConv**: Comparada con convolución estándar, DeformConv y GA-DeformConv, AlignConv supera en ~3% mAP con solo 1.41 GFLOPs adicionales. Especialmente mejora en categorías difíciles como bridge (+2%), small/large vehicles y helicopters.
- **Efectividad de ARN y ARF**: Los experimentos de ablación (Tabla III) muestran que ARN sola mejora 1.26% sobre baseline; ACL sola mejora 4.17%; ARF sola es irrelevante sin ACL; pero ACL+ARF juntos suben de 73.24% a 74.12%, mostrando que ARF augmenta información de orientación que ACL ya no puede capturar.
- **Diseño de red**: La configuración con FAM de 1 capa y ODM de 3 capas es óptima (74.12% mAP, 198 GFLOPs), y el rendimiento es insensible a la profundidad. Redes más profundas reducen el mAP en objetos pequeños por campo receptivo excesivo.

* **Problemas atacados**: Demostrar que cada componente aporta mejoras independientes y que el diseño final es el mejor balance entre velocidad y exactitud.
* **Limitaciones de ese entonces**: No existía un análisis sistemático de cómo cada componente de alineación contribuye en el contexto de imágenes aéreas orientadas.
* **Soluciones alcanzadas**: Los experimentos de ablación cuantifican la contribución de ARN (+1.26%), ACL (+4.17%), y la sinergia ACL+ARF (+0.88%), justificando el diseño final de S²A-Net.

#### D. Detecting on Large-Size Images

Se explora una estrategia alternativa: en lugar de recortar la imagen grande en chips pequeños (enfoque dominante en DOTA), inferir directamente sobre la imagen de tamaño original. Los experimentos (Tabla V) muestran que:

- Reducir el stride de recorte de 1024 a 512 mejora el mAP de 71.20% a 74.62% pero triplica el número de chips e incrementa el tiempo en 135%.
- Detectar sobre imágenes originales (sin recorte, 937 imágenes) logra 74.01% mAP con solo 120 segundos de inferencia (vs. 246 segundos con stride 824), una reducción del 50% en tiempo.
- Usando FAM para detección + FP16, se logra 70.85% mAP en 97 segundos.
- Comparado con ClusDet (que genera chips de clusters), S²A-Net procesa solo 458 imágenes y supera a ClusDet por un margen amplio.

* **Problemas atacados**: El overhead temporal y de memoria de procesar imágenes aéreas grandes mediante recorte en chips, y los problemas de detección en los bordes de los chips.
* **Limitaciones de ese entonces**: ClusDet intentaba solucionar esto generando chips de clusters, pero introducía operaciones complejas de generación y fusión de chips con caída de rendimiento significativa.
* **Soluciones alcanzadas**: S²A-Net puede detectar directamente en imágenes grandes con pérdida de exactitud negligible y reducción del 50% en tiempo de inferencia, gracias a su arquitectura completamente convolucional.

#### E. Comparisons with the State-of-the-Art

Se compara S²A-Net contra el estado del arte en DOTA y HRSC2016:

- **DOTA**: S²A-Net alcanza 74.01% mAP a 22.6 FPS con ResNet50-FPN (sin augmentación), y 76.11% con ResNet101-FPN, superando a todos los métodos de dos etapas y una etapa. Con multi-escala logra 79.42% (R50) y 79.15% (R101), siendo el mejor resultado reportado en el momento. Mejora especialmente en categorías difíciles: bridge, soccer-ball field, swimming pool, helicopter.
- **HRSC2016**: S²A-Net alcanza 90.17% (VOC2007) y 95.01% (VOC2012) mAP con **solo 1 anclaje**, superando a R3Det (89.26% con 21 anclajes) y CenterMapNet (92.8% con 15 anclajes), eliminando eficazmente la necesidad de anclajes orientados predefinidos.

* **Problemas atacados**: Demostrar que S²A-Net es el nuevo estado del arte en detección orientada de objetos aéreos, tanto en exactitud como en eficiencia.
* **Limitaciones de ese entonces**: Los mejores métodos previos requerían 20+ anclajes por ubicación o fusión de modelos para lograr alta exactitud, sacrificando velocidad.
* **Soluciones alcanzadas**: S²A-Net logra el mejor rendimiento con un solo anclaje cuadrado por ubicación, validando que la clave está en la alineación de características y no en el número de anclajes.

---

### V. Conclusion

La conclusión resume las contribuciones del artículo: S²A-Net es un detector simple y efectivo de una sola etapa para objetos orientados que logra alineación completa de características mediante FAM (AlignConv + ARN) y reduce la inconsistencia clasificación–localización mediante ODM (ARF + pooling invariante). Adicionalmente, la estrategia de detección en imágenes de tamaño original ofrece un mejor trade-off entre velocidad y exactitud. Los experimentos extensos en DOTA y HRSC2016 validan que S²A-Net alcanza el estado del arte en ambas métricas.

* **Problemas atacados**: Sintetizar y validar el impacto global del trabajo frente a la comunidad.
* **Limitaciones de ese entonces**: No se mencionan limitaciones futuras explícitas, aunque implícitamente el método depende de una definición de ángulo específica (θ ∈ [−π/4, 3π/4]) que puede causar discontinuidades angulares.
* **Soluciones alcanzadas**: S²A-Net establece un nuevo paradigma en detección orientada de una etapa que combina alineación de características explícita y características orientadas de manera eficiente y efectiva.
