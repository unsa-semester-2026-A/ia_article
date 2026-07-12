# Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models

- **Key**: Jocher2026
- **Year**: 2026
- **Venue**: arXiv (Ultralytics)

## Resumen
Ultralytics YOLO26 presenta una familia unificada de modelos de visión en tiempo real que soluciona limitaciones clave de los detectores YOLO previos (como YOLOv8 y YOLO11). Introduce un diseño de doble cabeza (dual-head) que permite inferencia nativa libre de NMS (NMS-free) mediante una rama "uno a uno" (one-to-one) y elimina por completo la pérdida de distribución focal (Distribution Focal Loss, DFL), aligerando la cabeza de regresión de cajas y eliminando restricciones de rango espacial. Su flujo de entrenamiento combina tres propuestas metodológicas coordinadas: el optimizador híbrido MuSGD (que integra Muon y SGD), una programación dinámica de pérdidas (Progressive Loss) para transferir gradualmente la supervisión hacia la cabeza de inferencia, y una estrategia de asignación de etiquetas robusta para objetos pequeños (STAL). Adicionalmente, YOLO26 introduce extensiones específicas para segmentación de instancias (fusión de prototipos multiescala y pérdida semántica auxiliar), estimación de pose (regresión probabilística de puntos clave vía RLE) y detección orientada (OBB con parametrización de lado largo y pérdida angular para objetos cuadrados). Finalmente, se presenta YOLOE-26, una variante de vocabulario abierto que alcanza 40.6 AP en LVIS minival.

## Secciones y Subsecciones

### 1. Introducción
Presenta el contexto de los detectores en tiempo real e identifica cuatro limitaciones fundamentales presentes en el estado del arte de la familia YOLO.
* **Problemas atacados**: Cuatro cuellos de botella: 1) Dependencia de NMS e infraentrenamiento de cabezas sin NMS. 2) Parámetros excesivos y rango acotado debido a DFL. 3) Ciclos largos de entrenamiento (600 épocas). 4) Ausencia de asignación positiva de anclas para objetos extremadamente pequeños en TAL.
* **Limitaciones de ese entonces**: Modelos basados en DETR en tiempo real (RT-DETR, D-FINE) requieren operadores personalizados o arquitecturas complejas de atención deformable que dificultan su exportación nativa a plataformas embebidas.
* **Soluciones alcanzadas**: Desarrollo de YOLO26, un framework unificado que conserva operadores convolucionales estándar y soporta 5 tareas de visión (detección, segmentación, pose, clasificación y OBB) distribuidas en 5 escalas de modelo (n/s/m/l/x) logrando un avance significativo en la frontera de Pareto precisión-latencia (Figura 1).

### 2. Trabajo Relacionado
Revisa la evolución de la detección basada en CNN (Two-Stage, One-Stage y NMS-Free), modelos basados en Transformers (DETR, RT-DETR), segmentación de instancias (Mask R-CNN, SOLOv2, YOLACT), estimación de pose (Stacked Hourglass, YOLO-Pose) y OBB (Circular Smooth Label, ProbIoU).

* **Problemas atacados**: Integración conceptual y posicionamiento de YOLO26 ante los diferentes desarrollos de visión por computador.
* **Limitaciones de ese entonces**: Falta de unificación en una sola base de código optimizada que combinara los aprendizajes de las metodologías open-vocabulary con la flexibilidad de despliegue embebido.
* **Soluciones alcanzadas**: Estructuración del pipeline YOLO26 para heredar los mecanismos de vocabulario abierto de YOLOE pero mejorando los codificadores de texto (MobileCLIP2), destilación por pseudolabeling y entrenamiento desacoplado de segmentación.

### 3. Metodología
Detalla las innovaciones en el diseño de arquitectura común, optimización de entrenamiento y las cabezas de tareas específicas.

* **Problemas atacados**: Reducción de costes de cómputo en la cabeza de regresión y mejora de la convergencia.
* **Limitaciones de ese entonces**: DFL expande la regresión de cajas de 4 escalares a $4K$ logits por píxel espacial (usualmente $K=16$), inflando innecesariamente el cómputo y limitando el tamaño máximo de objetos a $2(K-1) \times \text{stride}$ píxeles.
* **Soluciones alcanzadas**: Remoción de DFL, diseño de doble cabeza y formulación del flujo integrado de regularización de pérdidas.

#### 3.2 Diseño de Arquitectura

##### 3.2.1 Detección End-to-End Libre de NMS
* **Problemas atacados**: Suprimir la necesidad de NMS costoso en tiempo de inferencia sin forzar un único modo rígido de despliegue.
* **Limitaciones de ese entonces**: Diseños previos dual-head aplicaban pesos simétricos constantes, penalizando la especialización de la cabeza uno a uno.
* **Soluciones alcanzadas**: Implementación de una cabeza "uno a uno" (One-to-One Head, por defecto en inferencia) que decodifica directamente un máximo de 300 predicciones combinando una cascada de Task-Aligned Learning (TAL, topk=7 y topk2=1). Se mantiene opcionalmente una cabeza de predicción densa "uno a muchos" (One-to-Many Head, topk=10) para despliegues donde el NMS sea óptimo en hardware.

##### 3.2.2 Eliminación de Distribution Focal Loss (DFL)
* **Problemas atacados**: Exceso de parámetros y limitaciones en el rango de regresión de cajas grandes a resoluciones altas.
* **Limitaciones de ese entonces**: En YOLO11n, DFL representa el 12% de los parámetros globales y el 20% de los GFLOPs de la cabeza. Adicionalmente, limita el tamaño de caja en resoluciones de entrada superiores (ej. 1280).
* **Soluciones alcanzadas**: Sustitución de DFL por regresión directa con pérdida L1. Se demuestra cualitativamente (Figura 4) que la cabeza sin DFL delimita de manera más precisa y robusta los objetos masivos en imágenes de 1280 px al no tener acotado su soporte de regresión.

#### 3.3 Metodología de Entrenamiento

##### 3.3.1 Optimizador MuSGD
* **Problemas atacados**: Aceleración de la convergencia y optimización del acondicionamiento de gradiente en redes de detección.
* **Limitaciones de ese entonces**: SGD momentum clásico requiere cerca de 600 épocas de entrenamiento en COCO para estabilizar métricas de regresión.
* **Soluciones alcanzadas**: Uso de MuSGD, un optimizador híbrido que aplica el optimizador de LLMs "Muon" (que realiza ortogonalización de actualizaciones de momento mediante iteraciones de Newton-Schulz) para pesos convolucionales y lineales multidimensionales, y SGD clásico para parámetros unidimensionales (sesgos y escalas de normalización). Logra +0.4 mAP en 500 épocas frente a 600 épocas de SGD (Tabla 4).

##### 3.3.2 Pérdida Progresiva (Progressive Loss)
* **Problemas atacados**: Asimetría en la dificultad de optimización de las cabezas duales.
* **Limitaciones de ese entonces**: La cabeza uno a muchos (o2m) tiene gradientes más densos y es fácil de optimizar al inicio, mientras que la uno a uno (o2o) es restrictiva. Entrenarlas con pesos fijos es subóptimo.
* **Soluciones alcanzadas**: Un esquema de decaimiento dinámico de pesos $L_{\text{total}} = \alpha(t)L_{\text{o2m}} + (1 - \alpha(t))L_{\text{o2o}}$ donde $\alpha(t)$ decae linealmente desde (0.8, 0.2) al inicio hasta (0.1, 0.9) al final del entrenamiento (Tabla 5).

##### 3.3.3 Asignación de Etiquetas Sensible a Objetos Pequeños (STAL)
* **Problemas atacados**: Pérdida de señal de gradiente para objetos pequeños debido a la discretización del mapa de características.
* **Limitaciones de ese entonces**: El TAL clásico descarta anclas que no caen dentro de la caja geométrica del objeto. Para objetos menores al paso del stride mínimo ($s_{\text{min}}=8$), ningún centro de ancla califica, resultando en cero asignaciones positivas.
* **Soluciones alcanzadas**: STAL crea un contorno geométrico surrogate de tamaño ampliado Clamped al stride de nivel de escala superior ($s_{\text{ref}}=16$) únicamente durante la etapa de filtrado de candidatos de TAL (Ecuaciones 4-6). Esto garantiza que los objetos diminutos reciban candidatos positivos asignados, manteniendo la caja real para el cómputo final de la pérdida de regresión.

#### 3.4 Extensiones Específicas de Tareas

##### 3.4.1 Segmentación de Instancias
* **Problemas atacados**: Mejorar la calidad de las máscaras de segmentación basadas en prototipos.
* **Limitaciones de ese entonces**: YOLO11 segmenta proyectando prototipos únicamente desde el nivel de mayor resolución $X_1$, perdiendo relaciones semánticas gruesas.
* **Soluciones alcanzadas**: 1) Módulo Proto Multiescala que concatena proyecciones $1 \times 1$ upsampled de capas profundas a la resolución fina $X_1$ (Ecuaciones 8 y 9). 2) Una rama auxiliar de segmentación semántica supervisada por BCE+Dice que guía el aprendizaje del backbone durante el entrenamiento y se descarta en la inferencia (Tabla 8).

##### 3.4.2 Estimación de Pose
* **Problemas atacados**: Modelar la incertidumbre en la regresión directa de puntos clave corporales.
* **Limitaciones de ese entonces**: OKS loss no penaliza la dispersión de error ni diferencia articulaciones ambiguas u ocluidas.
* **Soluciones alcanzadas**: Integración de RLE (Residual Log-Likelihood Estimation). Se añade una rama que predice la desviación estándar $\sigma = (\sigma_x, \sigma_y)$ por articulación y estima la densidad residual bajo un flujo normalizado RealNVP. Esto reduce el peso de gradiente en articulaciones invisibles y estabiliza el mAP (+7.2 pose AP sobre YOLO11, ver Tabla 9).

##### 3.4.3 Detección Orientada (OBB)
* **Problemas atacados**: Discontinuidades angulares en límites de periodicidad e inestabilidad por intercambio de lados (edge swapping).
* **Limitaciones de ese entonces**: La convención OpenCV de ángulo agudo $(0, 90^\circ]$ intercambia altura y ancho cerca de los límites angulares, y ProbIoU se vuelve ciego al ángulo para cajas cuadradas (ancho $\approx$ alto).
* **Soluciones alcanzadas**: 1) Adopción de la definición de lado largo de MMRotate con rango $[-45^\circ, 135^\circ)$ restringiendo el ancho a ser mayor que la altura y prediciendo el ángulo directamente sin la no linealidad de sigmoid. 2) Diseño de una pérdida angular específica para objetos cuadrados penalizando desviaciones basadas en un multiplicador del ratio de aspecto (Ecuación 15).

#### 3.6 YOLOE-26: Detección y Segmentación de Vocabulario Abierto
* **Problemas atacados**: Generalizar detecciones a clases no vistas a alta velocidad.
* **Limitaciones de ese entonces**: YOLOE-v8/11 sufrían interferencia al entrenar la segmentación junto con el alineamiento de texto-región.
* **Soluciones alcanzadas**: Integración del backbone YOLO26 con codificador de texto MobileCLIP2, un motor de pseudoetiquetado guiado por un profesor con vocabulario de 4585 clases y entrenamiento de segmentación desacoplado secuencialmente del modelo base.

### 4. Experimentos
Muestra las validaciones en COCO (detección, segmentación, pose), DOTA-v1.0 (OBB) e ImageNet (clasificación).

* **Problemas atacados**: Validar la consistencia de YOLO26 frente a modelos previos bajo las mismas configuraciones.
* **Limitaciones de ese entonces**: Los resultados previos no desglosaban de manera controlada el impacto individual de cada mejora incremental de la arquitectura.
* **Soluciones alcanzadas**: Presentación de tablas detalladas de ablación por componentes (Tabla 2), tamaño de referencia de STAL (Tabla 6) y rendimiento open-vocabulary en LVIS (Tabla 12).

### 5. Conclusión
YOLO26 redefine la frontera de Pareto mAP-latencia en tiempo real en la GPU NVIDIA T4 (40.9-57.5 mAP). Demuestra que es viable prescindir de la complejidad de DFL y NMS a través de un diseño arquitectónico simplificado y un régimen de optimización cooperativo (MuSGD + Progressive Loss + STAL), sentando las bases para futuros detectores ligeros de vocabulario abierto e inferencia end-to-end unificada.
