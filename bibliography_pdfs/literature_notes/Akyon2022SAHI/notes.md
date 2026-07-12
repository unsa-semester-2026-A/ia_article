# Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection

- **Key**: Akyon2022SAHI
- **Year**: 2022
- **Venue**: ICIP

## Resumen
SAHI (Slicing Aided Hyper Inference) es un framework y pipeline de código abierto diseñado para mejorar sustancialmente la detección de objetos pequeños en imágenes de alta resolución (como las obtenidas por drones o satélites). SAHI propone un enfoque genérico aplicable sobre cualquier detector de objetos moderno (como YOLOv5, FCOS, VFNet o TOOD) sin necesidad de modificar su arquitectura. El framework consta de dos componentes clave: **Slicing Aided Inference (SAHI)**, que corta la imagen de entrada en parches solapados de menor tamaño, los escala para aumentar la resolución relativa de los objetos pequeños, realiza forward passes independientes y luego fusiona las predicciones mediante Non-Maximum Suppression (NMS); y **Slicing Aided Fine-tuning (SF)**, que aumenta el dataset de entrenamiento extrayendo parches de imágenes y escalándolos para entrenar el detector con objetos pequeños de mayor tamaño en píxeles. SAHI logra aumentos notables de hasta 14.5% AP de forma acumulativa en datasets retadores como VisDrone y xView, ofreciendo un excelente balance para entornos de producción, ya que incrementa el tiempo de cómputo linealmente pero mantiene el uso de memoria VRAM constante.

## Secciones y Subsecciones

### 1. Introduction
Presenta los desafíos de detectar objetos pequeños y lejanos en imágenes de alta resolución y propone el método de segmentación en parches (slicing) como solución universal y eficiente en memoria.
* **Problemas atacados**: La baja precisión en la detección de objetos pequeños en aplicaciones de videovigilancia y tomas aéreas, los cuales cubren una cantidad de píxeles muy reducida y carecen de características detalladas.
* **Limitaciones de ese entonces**: Los detectores se entrenan comúnmente con imágenes de baja resolución (como 640x480 de MS COCO) donde los objetos cubren el 60% de la altura de la imagen en promedio. Al procesar tomas aéreas o de alta resolución de drones (4K) donde los objetos cubren menos del 1% del ancho (criterio DORI), el rendimiento disminuye drásticamente. Además, procesar imágenes gigantescas directamente desborda la memoria VRAM de la GPU.
* **Soluciones alcanzadas**: Se propone SAHI, que procesa parches segmentados y redimensionados de la imagen para elevar la resolución efectiva de los objetos pequeños sin aumentar los requerimientos de memoria del sistema.

### 2. Related Work
Resume la literatura científica sobre detectores de una y dos etapas, detectores libres de anchors (anchor-free) y los enfoques específicos previos para objetos pequeños.
* **Problemas atacados**: La ineficacia de los detectores genéricos en imágenes aéreas de alta resolución y los inconvenientes de las soluciones ad-hoc previas para objetos pequeños.
* **Limitaciones de ese entonces**:
  * Métodos que modifican la arquitectura o las funciones de pérdida (como PBLS) impiden aprovechar los pesos preentrenados e imponen entrenamientos costosos desde cero.
  * El aumento de datos por copia de objetos pequeños requiere anotaciones de segmentación a nivel de píxel que rara vez existen en datasets de bounding boxes.
  * Técnicas de súper-resolución (como redes GAN) o selección de regiones de atención requieren redes adicionales pesadas, incrementando el coste computacional.
  * Los métodos de slicing previos no eran modulares y se ataban a detectores específicos.
* **Soluciones alcanzadas**: SAHI se presenta como un pipeline modular, de código abierto y agnóstico al modelo, compatible con librerías populares como MMDetection, Detectron2 y YOLOv5 sin alterar sus estructuras internas.

### 3. Proposed Approach
Describe las dos fases de la arquitectura propuesta: el ajuste fino por rebanado (SF) y la inferencia asistida por rebanado (SAHI).
* **Problemas atacados**: Los altos requerimientos de memoria al entrenar o evaluar con mapas de características gigantescos.
* **Limitaciones de ese entonces**: Tratar de detectar objetos pequeños escalando toda la imagen incrementa el coste de memoria de forma cuadrática.
* **Soluciones alcanzadas**: Dividir el flujo de procesamiento espacial en parches locales que se manejan individualmente con resoluciones estándar para el detector.

#### Slicing Aided Fine-tuning (SF)
Detalla el aumento de datos durante el entrenamiento mediante la segmentación de la imagen en parches de escala ampliada.
* **Problemas atacados**: La falta de generalización de los detectores preentrenados en COCO ante objetos aéreos pequeños.
* **Limitaciones de ese entonces**: Modelos preentrenados en datasets estándar rinden mal en tomas de drones ya que no están acostumbrados a procesar objetos de unos pocos píxeles de altura.
* **Soluciones alcanzadas**: Las imágenes del dataset de entrenamiento se dividen en parches con dimensiones de ancho y alto aleatorias dentro de rangos predefinidos ($[M_{min}, M_{max}]$ y $[N_{min}, N_{max}]$). Estos parches se redimensionan a tamaños de 800 a 1333 píxeles, logrando que los objetos pequeños tengan dimensiones relativamente grandes durante el ajuste fino. Estos parches se mezclan con las imágenes completas originales para entrenar el modelo.

#### Slicing Aided Hyper Inference (SAHI)
Describe la tubería de inferencia segmentada y el post-procesamiento para fusionar los bounding boxes predichos.
* **Problemas atacados**: La incapacidad del detector para disparar activaciones robustas en objetos pequeños durante la inferencia estándar.
* **Limitaciones de ese entonces**: Si se pasa la imagen completa de alta resolución al detector, el submuestreo de la red elimina las características de los objetos pequeños antes de llegar a las capas de predicción.
* **Soluciones alcanzadas**: La imagen de entrada se corta en $l$ parches de tamaño $M \times N$ con solapamiento. Se corre el detector de forma independiente sobre cada parche redimensionado. De manera opcional, se corre una inferencia sobre la imagen completa (Full Inference - FI) para no perder objetos grandes. Finalmente, todas las predicciones se proyectan a las coordenadas globales de la imagen original y se unifican mediante NMS con un umbral de emparejamiento $T_m$ y de filtrado de confianza $T_d$.

### 4. Results
Presenta las pruebas en VisDrone y xView, utilizando detectores como FCOS, VFNet y TOOD.
* **Problemas atacados**: Validación experimental del framework y análisis de tipos de error en objetos pequeños.
* **Limitaciones de ese entonces**: Los datasets aéreos y satelitales son sumamente complejos debido a la densidad de objetos y el fuerte desbalance de clases (por ejemplo, xView tiene 60 clases altamente desbalanceadas).
* **Soluciones alcanzadas**:
  * SAHI sin reentrenamiento eleva el AP en VisDrone en +6.8% (FCOS), +5.1% (VFNet) y +5.3% (TOOD).
  * Con ajuste fino (SF) y SAHI conjunto, el AP acumulado sube en +12.7% (FCOS), +13.4% (VFNet) y +14.5% (TOOD).
  * En xView, donde el entrenamiento estándar obtiene apenas ~2.1% AP50 por la pequeñez de los objetos, la combinación SF + SAHI eleva el AP50 hasta 20.6%.
  * Se demuestra que un solapamiento (Patch Overlap - PO) del 25% reduce significativamente los falsos negativos en los bordes de los parches.

### 5. Conclusion
Resume los resultados y proyecta el trabajo hacia segmentación de instancias.
* **Problemas atacados**: Viabilidad práctica de SAHI en entornos embebidos u ordenadores con recursos de hardware limitados.
* **Limitaciones de ese entonces**: Los métodos para mejorar la detección en alta resolución suelen requerir GPUs con cantidades prohibitivas de VRAM.
* **Soluciones alcanzadas**: SAHI escala el tiempo de cómputo de manera lineal según el número de parches, pero mantiene los requisitos de memoria de GPU fijos y acotados. Esto permite ajustar dinámicamente los tamaños de parches para balancear el tiempo de cómputo y la memoria según el hardware disponible.
