# ColabNAS: Obtaining lightweight task-specific convolutional neural networks following Occam’s razor

- **Key**: Garavagno2022ColabNAS
- **Year**: 2024
- **Venue**: Future Generation Computer Systems

## Resumen
ColabNAS es una técnica económica de búsqueda de arquitectura neuronal consciente del hardware (HW NAS) para producir redes neuronales convolucionales (CNN) ligeras y específicas para una tarea. El artículo critica la tendencia habitual de aplicar aprendizaje por transferencia (Transfer Learning, TL) a partir de redes grandes preentrenadas (como MobileNetV2), argumentando que esto es excesivo (*overkill*) para tareas específicas donde se dispone de suficientes datos para entrenar desde cero. Para democratizar el acceso a HW NAS (que tradicionalmente requiere cientos de horas de GPU y supercomputadoras), ColabNAS propone un espacio de búsqueda regular tipo VGG de celdas apiladas y una estrategia de búsqueda libre de derivadas inspirada en la navaja de Ockham. Esta estrategia incrementa o reduce las celdas y los canales del primer nivel secuencialmente basándose en si la capacidad de generalización mejora y si se respetan las restricciones físicas (RAM, Flash, MACC). Los resultados muestran que ColabNAS obtiene modelos que compiten con el estado del arte (como MCUNet y MicroNets) en el dataset Visual Wake Words (VWW) en solo 3.1 horas de GPU, permitiendo su ejecución en servicios gratuitos como Google Colab o Kaggle.

## Secciones y Subsecciones

### 1. Introducción
Presenta el surgimiento de dispositivos portátiles y sensores inteligentes distribuidos que requieren CNNs ligeras. Discute el contraste entre el diseño manual desde cero, HW NAS tradicional y el aprendizaje por transferencia (TL).
* **Problemas atacados**: La alta barrera de entrada para crear CNNs personalizadas optimizadas para microcontroladores debido al costo computacional de HW NAS o al conocimiento experto requerido para el diseño manual.
* **Limitaciones de ese entonces**: Los métodos HW NAS líderes de la época (como MCUNet o MicroNets) consumen alrededor de 300 horas de GPU para buscar modelos eficientes. Por ello, los desarrolladores suelen recurrir al aprendizaje por transferencia (TL) usando modelos sobredimensionados (como MobileNetV2 o EfficientNet) que, aunque rápidos de adaptar, tienen un costo de RAM, Flash y cómputo excesivo para microcontroladores de bajos recursos.
* **Soluciones alcanzadas**: Propuesta de ColabNAS, un algoritmo de bajo costo de búsqueda de arquitectura que encuentra redes ultraligeras desde cero en pocas horas de GPU de uso gratuito.

### 2. Trabajos Relacionados
Revisa los benchmarks establecidos para TinyML (como VWW) y la evolución de HW NAS (desde aprendizaje por refuerzo hasta enfoques basados en superredes y optimización de gradiente).
* **Problemas atacados**: La optimización automática de hiperparámetros de red neuronal (NAS) bajo restricciones estrictas de hardware en el dispositivo.
* **Limitaciones de ese entonces**: Los primeros enfoques basados en aprendizaje por refuerzo (como MNASNet) tardan más de 40,000 horas de GPU. Enfoques más recientes reducen esto entrenando una "superred" de la cual extraen subredes (como TinyNAS de MCUNet), pero aun así requieren hardware de alto rendimiento y cientos de horas de cómputo para entrenamiento de base.
* **Soluciones alcanzadas**: ColabNAS propone limitar la exploración de manera inteligente sin construir superredes gigantescas, reduciendo el espacio de búsqueda a estructuras regulares y compactas.

### 3. ColabNAS
Describe el núcleo técnico del framework propuesto dividiéndolo en tres pilares: espacio de búsqueda, formulación del problema y estrategia de búsqueda.

#### 3.1. Espacio de Búsqueda y Formulación del Problema
Detalla la estructura matemática de las soluciones candidatas y los límites impuestos por las restricciones de hardware.
* **Problemas atacados**: La definición matemática de los límites de búsqueda para asegurar la viabilidad de despliegue en microcontroladores con recursos críticos (por ejemplo, 20-40 kiB de RAM).
* **Limitaciones de ese entonces**: Los espacios de búsqueda excesivamente amplios o heterogéneos requieren explorar millones de combinaciones y dificultan la estimación simple de requerimientos físicos en microcontroladores.
* **Soluciones alcanzadas**: Definición de un espacio de búsqueda celular que comienza con una capa convolucional 2D y añade secuencialmente celdas compuestas por una capa de pooling y otra convolucional. La duplicación de canales se modula con una fórmula decreciente para evitar el crecimiento exponencial de parámetros. El problema se formula como una optimización donde se maximiza la precisión de validación sujeta a límites máximos de RAM, Flash y operaciones de multiplicación-acumulación (MACC). La parada del apilamiento de celdas se rige por la navaja de Ockham: no multiplicar entidades (capas) más allá de lo necesario si no mejora la generalización.

#### 3.2. Detalles de la Arquitectura de la Red
Presenta las decisiones de diseño estables de las celdas del modelo.
* **Problemas atacados**: Asegurar la robustez y capacidad de convergencia de arquitecturas generadas de manera automática.
* **Limitaciones de ese entonces**: Convoluciones complejas o ramificadas pueden ser difíciles de compilar de forma eficiente para microcontroladores simples.
* **Soluciones alcanzadas**: Adopción de un diseño regular basado en VGG16: convoluciones con kernels de $3\times3$, zero padding para preservar tamaño y pooling de $2\times2$. Se usa Global Average Pooling 2D al final del bloque convolucional para reducir dimensiones, seguido de una capa densa intermedia y la capa de clasificación final. Se integra estandarización min-max, Batch Normalización y aumento de datos para acelerar la convergencia.

#### 3.3. Estrategia de Búsqueda
Detalla el algoritmo libre de derivadas que recorre el espacio en dos fases alternas de optimización por coordenadas.
* **Problemas atacados**: Encontrar la arquitectura óptima minimizando la cantidad de entrenamientos completos requeridos.
* **Limitaciones de ese entonces**: Los algoritmos evolutivos o de refuerzo tradicionales exploran el espacio de manera aleatoria u homogénea, desperdiciando tiempo en entrenar configuraciones inviables o sobredimensionadas.
* **Soluciones alcanzadas**: Implementación de una búsqueda alterna por ejes (número de canales de la primera capa $k$ y número de celdas $c$). Primero, para un canal fijo, el algoritmo añade celdas una a una (dirección de celdas $d=(0,1)$) hasta que la precisión de validación deje de mejorar o se violen restricciones físicas. Luego, desplaza el punto de partida duplicando los canales iniciales si el rendimiento mejora, o dividiéndolos a la mitad en caso contrario.

### 4. Métodos Experimentales
Presenta los cinco problemas de clasificación para evaluar el algoritmo y la plataforma de pruebas (Google Colab con GPU Tesla T4).
* **Problemas atacados**: Validación experimental de ColabNAS en tareas específicas de diversos tamaños y tipologías.
* **Limitaciones de ese entonces**: Falta de reproducibilidad en investigaciones NAS debido a dependencias de clusters privados de supercomputación.
* **Soluciones alcanzadas**: Selección de 5 datasets públicos (Melanoma, Visual Wake Words, Animals-3, Flowers-4 y MNIST) y definición de un entorno público y gratuito en Google Colab con GPU Tesla T4 para demostrar la accesibilidad del método.

### 5. Comparación con Redes Obtenidas por Aprendizaje por Transferencia
Evalúa ColabNAS comparando sus resultados con MobileNetV2 preentrenada en ImageNet en tareas específicas.
* **Problemas atacados**: El sobredimensionamiento y desperdicio de recursos provocado por el uso sistemático de transfer learning (TL).
* **Limitaciones de ese entonces**: MobileNetV2 tiene requerimientos fijos de RAM y Flash que exceden los límites de microcontroladores pequeños, limitando el despliegue de TinyML en hardware modesto.
* **Soluciones alcanzadas**: 
  - Para Melanoma, ColabNAS supera a TL en precisión (+3.1%) consumiendo 4.6 veces menos RAM y 35.2 veces menos Flash.
  - Para Animals-3 y Flowers-4, ColabNAS reduce drásticamente la ocupación de Flash (hasta 13.4 y 44.4 veces menos, respectivamente) y RAM (2.6 y 4.6 veces menos) a cambio de una pérdida aceptable de precisión (5.4% - 6%).

### 6. Evaluación de la Capacidad Consciente del Hardware (Hardware-Aware)
Valida la capacidad de ColabNAS para adaptar automáticamente la arquitectura a las restricciones específicas de tres microcontroladores STMicroelectronics (L0, L1, L4) con 20 kiB, 32 kiB y 40 kiB de RAM.
* **Problemas atacados**: Adaptar la topología del modelo a diferentes chips de microcontroladores sin rediseño manual.
* **Limitaciones de ese entonces**: Un modelo TinyML genérico suele requerir reajustes empíricos a base de ensayo y error para lograr caber en la memoria de un chip específico.
* **Soluciones alcanzadas**: ColabNAS encuentra con éxito un modelo ejecutable para cada chip en todas las tareas, optimizando el tamaño según el presupuesto del microcontrolador. Se observa que el tamaño del modelo final está fuertemente acotado por las restricciones de hardware y permanece relativamente constante entre diferentes datasets para una misma especificación física de MCU.

### 7. Comparación con Técnicas HW NAS del Estado del Arte en el Dataset Visual Wake Words
Compara a ColabNAS con MCUNet y MicroNets en la tarea estándar TinyML.

#### 7.1. Comparación de Costos de Búsqueda
Analiza la eficiencia de cómputo del proceso de búsqueda.
* **Problemas atacados**: Reducir el enorme gasto energético y económico requerido para la búsqueda de modelos TinyML.
* **Limitaciones de ese entonces**: MCUNet requiere 300 horas de GPU, y MicroNets utiliza DNAS durante 200 épocas sobre superredes complejas, resultando en días de procesamiento GPU.
* **Soluciones alcanzadas**: ColabNAS encuentra su solución óptima en solo 3.1 horas de GPU (aproximadamente 100 veces más rápido que MCUNet). En el hardware STM32F446RE, ColabNAS supera al modelo de MicroNets en todas las métricas (precisión, RAM, Flash y latencia). MCUNet retiene mayor precisión (+9.8%) pero a cambio de consumir 5.35 veces más RAM, 25.47 veces más Flash y ser 5 veces más lento en inferencia.

### 8. Conclusión
Sintetiza las aportaciones y concluye sobre el impacto del trabajo en sostenibilidad y democratización.
* **Problemas atacados**: La necesidad de democratizar el desarrollo de IA y mejorar la sostenibilidad energética de los sistemas TinyML.
* **Limitaciones de ese entonces**: La investigación en NAS estaba restringida a organizaciones con presupuestos masivos para servidores.
* **Soluciones alcanzadas**: ColabNAS demuestra que una estrategia simple inspirada en la navaja de Ockham puede reducir drásticamente el coste de búsqueda de arquitecturas TinyML eficientes a niveles accesibles para cualquier investigador a través de plataformas gratuitas, reduciendo simultáneamente el coste de almacenamiento y el consumo energético en la inferencia física sobre wearables y sensores distribuidos.
