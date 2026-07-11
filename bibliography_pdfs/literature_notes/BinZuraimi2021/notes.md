# Vehicle Detection and Tracking using YOLO and DeepSORT

- **Key**: BinZuraimi2021
- **Year**: 2021
- **Venue**: ISCAIE (IEEE Symposium on Computer Applications & Industrial Electronics)

## Resumen
Este artículo presenta el desarrollo y la evaluación de un sistema inteligente de monitoreo de tráfico para la detección, clasificación y conteo automático de vehículos en carreteras de Malasia. Debido al incremento constante del volumen de vehículos y la congestión vehicular asociada, se requiere una gestión de tráfico más inteligente. El sistema propuesto utiliza el algoritmo de detección de una sola etapa YOLOv4 integrado con el algoritmo de seguimiento multi-objeto DeepSORT en la plataforma TensorFlow. El sistema es capaz de detectar y clasificar vehículos en cuatro categorías principales (automóvil, motocicleta, autobús y camión) y realizar un conteo automatizado cuando cruzan una línea de interés virtual en el video. YOLOv4 logró el mejor rendimiento con un 82.08% de mAP@0.5 en un dataset personalizado de 7319 imágenes, a una velocidad de inferencia de 14.12 FPS en una GPU local GTX 1660ti, mientras que su versión ligera YOLOv4-tiny alcanzó 40.11 FPS con 76.14% de mAP@0.5, presentándose como una alternativa viable para hardware limitado.

## Secciones y Subsecciones

### I. Introduction
Establece la importancia del monitoreo de tráfico debido al crecimiento acelerado del parque automotor en Malasia (con 31.2 millones de vehículos registrados en 2019) y las consecuencias negativas de la congestión.
* **Problemas atacados**: La ineficiencia en la gestión y análisis del tráfico vehicular y el riesgo de accidentes y congestión en las autopistas.
* **Limitaciones de ese entonces**: Contar y clasificar vehículos de forma manual requiere operadores humanos que deben vigilar múltiples pantallas constantemente. Esto genera fatiga rápida, baja precisión y la imposibilidad de procesar flujos masivos de video en tiempo real.
* **Soluciones alcanzadas**: Implementación de un sistema automatizado de visión computacional y aprendizaje profundo (Deep Learning) que detecta, clasifica y realiza el conteo de vehículos de manera continua e interactiva.

### II. Related Work
Revisa los métodos de visión computacional tradicionales y los avances recientes en redes neuronales convolucionales.
* **Problemas atacados**: La baja precisión y el alto costo computacional asociados a los algoritmos de detección vehicular antiguos.
* **Limitaciones de ese entonces**: Los métodos tradicionales basados en características hechas a mano como HOG (Histogram of Oriented Gradients) y Haar-like features dependían de sustraer el fondo en movimiento y daban una tasa muy alta de falsos positivos en condiciones cambiantes. Por su parte, los detectores de aprendizaje profundo de dos etapas (ej. R-CNN o Faster R-CNN) eran precisos pero demasiado lentos para aplicaciones viales de tiempo real.
* **Soluciones alcanzadas**: Se introdujeron los detectores de una sola etapa de la familia YOLO (You Only Look Once), que transforman la localización de cajas en un problema de regresión directa a nivel de red neuronal convolucional (CNN), optimizando drásticamente la velocidad.

### III. Methodology
Describe el pipeline completo del sistema inteligente de tráfico, desde la descarga de datos hasta la inferencia final.
* **Problemas atacados**: La dificultad de integrar pipelines de entrenamiento complejos basados en GPU con sistemas locales de inferencia y despliegue en sistemas de cámaras viales.
* **Limitaciones de ese entonces**: Configurar el entorno de Darknet nativo en C/CUDA puede ser inestable y complejo en computadoras personales basadas en Windows.
* **Soluciones alcanzadas**: Se estructuró un flujo de trabajo que recopila imágenes viales automáticamente, las etiqueta en formato de anotación YOLO, entrena el modelo en la nube y lo despliega localmente en TensorFlow sobre Python.

#### A. Installation
Instalación de las dependencias de software del proyecto.
* **Problemas atacados**: La incompatibilidad de sistemas y scripts diseñados originalmente para entornos de tipo Unix.
* **Limitaciones de ese entonces**: La mayoría del código y las utilidades del repositorio de YOLOv4 se basan en Linux/macOS, dificultando su ejecución directa en Windows 10.
* **Soluciones alcanzadas**: Instalación de GitBash para proveer una terminal Linux en Windows, y el uso del gestor de paquetes Conda para configurar ambientes de GPU aislados (con TensorFlow-GPU, OpenCV, pillow, etc.).

#### B. Collection of the images
El proceso de descarga y recolección de los conjuntos de datos.
* **Problemas atacados**: La escasez y el costo en tiempo de recopilar manualmente imágenes de entrenamiento de múltiples tipos de vehículos en diferentes poses.
* **Limitaciones de ese entonces**: Descargar imágenes de forma individual en internet consume semanas y carece de uniformidad y metadatos de anotación.
* **Soluciones alcanzadas**: Uso del kit de herramientas OIDv4 para descargar automáticamente miles de imágenes categorizadas de vehículos directamente del repositorio de imágenes abiertas de Google.

#### C. Labelling and classifying
El proceso de anotación y asignación de etiquetas de las cajas delimitadoras.
* **Problemas atacados**: La alta inversión de tiempo necesaria para etiquetar manualmente coordenadas de cajas en formato de texto.
* **Limitaciones de ese entonces**: Escribir de forma manual los archivos `.txt` en el formato estructurado de YOLOv4 (`<object-class> <x> <y> <width> <height>`) es propenso a errores humanos de escala o posición.
* **Soluciones alcanzadas**: OIDv4 Toolkit automatiza este proceso generando directamente los archivos de anotaciones en el formato y la carpeta correspondientes al descargar los datos.

#### D. Training YOLO model
El entrenamiento de los pesos de la red profunda en la nube.
* **Problemas atacados**: La limitación computacional de entrenar modelos pesados en computadoras personales estándar.
* **Limitaciones de ese entonces**: Entrenar un modelo YOLOv4 en una GPU de escritorio de rango medio (como una GTX 1660ti) requiere más de 8 horas de cómputo para lograr la convergencia de 6000 iteraciones.
* **Soluciones alcanzadas**: Uso de Google Colab para ejecutar el entrenamiento sobre GPUs de alto rendimiento (Tesla T4 de 16GB) de forma gratuita en la nube, reduciendo el tiempo de entrenamiento a la mitad (4 horas).

#### E. Run code in GitBash
La conversión del formato de pesos para el despliegue local.
* **Problemas atacados**: La incompatibilidad de la infraestructura de desarrollo en C de Darknet con el framework de inferencia en Python en Windows.
* **Limitaciones de ese entonces**: La biblioteca original Darknet requiere compiladores de C complejos y es poco amigable para integrarse con librerías de tracking en Python.
* **Soluciones alcanzadas**: Se convirtieron los pesos `.weights` de Darknet a archivos de modelo nativos de TensorFlow (utilizando utilidades de DarkFlow), lo que facilita su inferencia ágil a través del script de Python local.

#### F. Python program development
Implementación del script de Python que integra la inferencia y el seguimiento temporal.
* **Problemas atacados**: La imposibilidad de contar y rastrear de forma persistente vehículos individuales a lo largo del tiempo.
* **Limitaciones de ese entonces**: Los detectores como YOLO solo detectan objetos en cuadros estáticos aislados, careciendo de memoria temporal. Esto causa duplicaciones en el conteo cuando los vehículos se ocluyen parcialmente por otros autos o infraestructura.
* **Soluciones alcanzadas**: Integración de DeepSORT, que extiende el algoritmo SORT mediante un Filtro de Kalman (para predicción de trayectoria), el algoritmo Húngaro (para emparejar la distancia IoU de cajas) y un descriptor de características visuales profundo para evitar que el ID del vehículo cambie tras una oclusión.

#### G. Output video
Generación y renderizado del flujo de video final procesado.
* **Problemas atacados**: Proporcionar una visualización intuitiva y auditable de los resultados del conteo e identificación de tráfico.
* **Limitaciones de ese entonces**: Los reportes viales numéricos planos carecen de soporte visual para verificar si los autos fueron contados correctamente o si hubo falsos positivos.
* **Soluciones alcanzadas**: Renderizado de un video final que superpone cajas delimitadoras de vehículos con su ID asignado, el contador de FPS en la esquina y una línea de cruce virtual que suma dinámicamente cada unidad que la atraviesa.

### IV. Results and Discussion
Presentación y análisis del desempeño comparativo de los modelos YOLO evaluados.
* **Problemas atacados**: La toma de decisiones técnicas sobre qué variante de detector de una etapa es la más adecuada para el despliegue práctico.
* **Limitaciones de ese entonces**: No existía una comparación controlada bajo el mismo hardware y dataset custom para el flujo vial de Malasia.
* **Soluciones alcanzadas**: Se evaluaron y documentaron los pesos, el mAP y la velocidad (FPS) de YOLOv3, YOLOv3-tiny, YOLOv4 y YOLOv4-tiny.

#### A. Datasets
Detalles del tamaño y división del dataset personalizado.
* **Problemas atacados**: El entrenamiento y validación de las 4 clases de vehículos elegidas.
* **Limitaciones de ese entonces**: Mezclar datos de prueba y entrenamiento genera evaluaciones sesgadas (data leakage).
* **Soluciones alcanzadas**: Recolección de 7319 imágenes de entrenamiento y un conjunto de validación separado con una relación del 30% (750 imágenes por clase para calcular mAP).

#### B. Weight
El tamaño físico de los modelos resultantes del entrenamiento.
* **Problemas atacados**: La restricción de almacenamiento físico en hardware embebido compacto.
* **Limitaciones de ese entonces**: Los modelos YOLOv4 y YOLOv3 completos pesan alrededor de 250MB, tamaño excesivo para chips de almacenamiento reducidos (como placas Raspberry Pi o móviles).
* **Soluciones alcanzadas**: Se demostró que las versiones "tiny" de YOLOv4 y YOLOv3 pesan solo 22MB y 33MB respectivamente, requiriendo 10 veces menos almacenamiento.

#### C. Mean Average Precision
Evaluación cuantitativa de la precisión de detección de los modelos.
* **Problemas atacados**: La medición exacta del error de localización y clasificación.
* **Limitaciones de ese entonces**: Calcular el mAP en conjuntos de datos oficiales masivos (COCO de 1GB) no refleja la precisión de un modelo ajustado localmente con un dataset más pequeño de 250MB.
* **Soluciones alcanzadas**: Se evaluaron los modelos con un umbral de IoU de 0.5 (Pascal VOC). YOLOv4 custom logró el mAP más alto con 82.08%, seguido por YOLOv3 (80.32%), YOLOv4-tiny (76.14%) y YOLOv3-tiny (66.03%).

#### D. Performance of model
Evaluación de la velocidad de procesamiento de video en tiempo real.
* **Problemas atacados**: La ralentización de la detección al procesar videos continuos en hardware local.
* **Limitaciones de ese entonces**: Los modelos pesados muy precisos (YOLOv4) degradan la velocidad por debajo de los 15 FPS, limitando su uso en sistemas interactivos de tráfico rápido.
* **Soluciones alcanzadas**: En una GPU GTX 1660ti, YOLOv3-tiny alcanzó 52.77 FPS y YOLOv4-tiny llegó a 40.11 FPS, mientras que YOLOv4 se redujo a 14.12 FPS. Se concluyó que YOLOv4-tiny ofrece el mejor equilibrio velocidad/precisión para su uso práctico.

### V. Conclusions
Resumen de hallazgos y sugerencias de arquitectura futura.
* **Problemas atacados**: La escalabilidad y descentralización del sistema inteligente de tráfico propuesto.
* **Limitaciones de ese entonces**: Alojar una computadora de escritorio con GPU GTX en cada poste de cámara vial en autopistas es costoso e inviable físicamente.
* **Soluciones alcanzadas**: Se recomendó el despliegue del software en dispositivos embebidos Raspberry Pi en postes locales y el uso de computación en la nube para procesar de forma centralizada y escalable los videos viales, enviando solo datos comprimidos para alertar de la congestión.
