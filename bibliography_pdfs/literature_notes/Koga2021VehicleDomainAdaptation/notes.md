# Adapting Vehicle Detector to Target Domain by Adversarial Prediction Alignment

- **Key**: Koga2021VehicleDomainAdaptation
- **Year**: 2021
- **Venue**: IGARSS

## Resumen
Este artículo aborda el problema de la degradación del rendimiento en detectores de objetos basados en Deep Learning aplicados a imágenes satelitales, originado por la discrepancia de distribución de datos (domain shift) entre el dominio de entrenamiento (source domain) y el de prueba (target domain). Los métodos de Adaptación de Dominio (DA) tradicionales se enfocan únicamente en alinear el extractor de características, lo que deja diferencias residuales que el clasificador final no puede procesar correctamente. Los autores proponen una novedosa técnica de **alineación adversaria de predicciones (Adversarial Prediction Alignment)** que adapta directamente el detector al dominio objetivo. El método alinea de manera simultánea en el espacio de salida tanto las predicciones del regresor de localización (offsets) como las del clasificador de categorías (confianza de clases). Para solucionar el desbalance extremo de clases típico en imágenes satelitales (donde el fondo predomina sobre los vehículos pequeños), se introduce la **Normalización de Pesos de Clase (Class Weight Normalization - CWN)**, la cual equilibra los gradientes adversarios de fondo y vehículos. Evaluada con el dataset satelital COWC como dominio origen y fotos aéreas de Japón como destino sobre un detector SSD-VGG16, la propuesta logra incrementar el AP en más de 5% frente al baseline de alineación exclusiva de características, demostrando la importancia de alinear las predicciones en el espacio de salida.

## Secciones y Subsecciones

### 1. Introduction
Presenta los fundamentos de la detección de vehículos en teledetección y las limitaciones de los métodos tradicionales de adaptación de dominio basados en características.
* **Problemas atacados**: La pérdida drástica de precisión de los detectores satelitales de vehículos al procesar imágenes de áreas geográficas, sensores o condiciones ambientales no vistas en el dataset de entrenamiento.
* **Limitaciones de ese entonces**: Los métodos de DA convencionales solo alinean las características (feature alignment), dejando discrepancias residuales que el clasificador original no puede absorber. Técnicas adaptativas en la salida (como la minimización de entropía o el alineamiento de incertidumbre) se diseñaron para tareas de clasificación de imágenes o segmentación de píxeles, resultando inadecuadas para la detección de objetos donde el modelo debe predecir simultáneamente clases discretas y coordenadas espaciales continuas.
* **Soluciones alcanzadas**: Proponer un método de alineación adversaria en el espacio de salida (prediction alignment) que entrena discriminadores dedicados a evaluar conjuntamente la localización espacial y la confianza categórica de las cajas delimitadoras.

### 2. Methodology
Describe el pipeline metodológico compuesto por el detector SSD, la alineación en cascada de características y predicciones, y el mecanismo de normalización de clases.
* **Problemas atacados**: Estabilización del entrenamiento de adaptación de dominio para detectores del tipo Single-Shot en entornos de clases altamente desequilibradas.
* **Limitaciones de ese entonces**: Intentar alinear el espacio de salida en detectores satelitales suele colapsar debido a que los anchors vacíos del fondo dominan los gradientes en el entrenamiento adversario.
* **Soluciones alcanzadas**: Diseñar un pipeline híbrido que realiza una primera alineación local de características y una posterior alineación global de predicciones con escalado adaptativo de pesos de clase.

#### 2.1. Vehicle Detector
* **Problemas atacados**: Elección de un framework de detección robusto y de bajo coste.
* **Limitaciones de ese entonces**: Los detectores de dos etapas (como Faster R-CNN) introducen un RPN y módulos de RoIAlign que fragmentan los mapas y dificultan el acoplamiento directo de discriminadores adversarios en la predicción final.
* **Soluciones alcanzadas**: Se selecciona un detector SSD (Single Shot MultiBox Detector) plano con VGG-16, el cual predice directamente en cada ubicación offsets de regresión $(cx, cy, w, h)$ y probabilidades de clase (fondo vs. vehículo) sobre anchors por defecto.

#### 2.2. Prediction Alignment
Describe la formulación del entrenamiento adversario para acoplar las predicciones en ambos dominios.
* **Problemas atacados**: Inestabilidad y oscilación de los discriminadores adversarios de salida.
* **Limitaciones de ese entonces**: La alineación adversaria en la salida falla si los extractores de características de ambos dominios difieren demasiado inicialmente.
* **Soluciones alcanzadas**: Aplicar una alineación en dos etapas consecutivas: alineación de características y alineación de predicciones.

##### 2.2.1. Feature alignment
* **Problemas atacados**: Generación de un espacio de características latentes común entre dominios.
* **Limitaciones de ese entonces**: Las características de alto nivel espacialmente variables impiden la convergencia de la pérdida adversaria del detector.
* **Soluciones alcanzadas**: Se define un discriminador de características $D_f$ que opera sobre parches locales de $3\times3$ píxeles en el mapa de características más somero de SSD. El extractor de características $M$ se entrena de forma adversaria para engañar a $D_f$.

##### 2.2.2. Prediction alignment
* **Problemas atacados**: Desalineación espacial y de confianza residual en el detector de destino.
* **Limitaciones de ese entonces**: Los alineadores tradicionales ignoran los gradientes espaciales de la regresión de cajas, limitándose a alinear probabilidades de clase.
* **Soluciones alcanzadas**: Se concatenan las salidas del regresor de localización y del clasificador en un único vector de predicción por anchor. Un discriminador de predicción $D_p$ se entrena adversariamente para evaluar y uniformizar la distribución de estos vectores entre ambos dominios.

##### 2.2.3. Class weight normalization (CWN)
* **Problemas atacados**: Desbalance de clases extremo en imágenes aéreas donde los vehículos pequeños ocupan una fracción mínima frente a la clase de fondo.
* **Limitaciones de ese entonces**: Durante la adaptación de dominio adversaria de las predicciones, los gradientes propagados desde los anchors del fondo son masivos y silencian a los de los vehículos pequeños, impidiendo que el detector aprenda a localizarlos correctamente en el nuevo dominio.
* **Soluciones alcanzadas**: Se introduce CWN, que calcula en cada minibatch un peso de clase $B = (b_0, b_1)$ inversamente proporcional a la frecuencia de predicciones asignadas a cada clase. Estos pesos equilibran la contribución de los vehículos ($b_1$) y el fondo ($b_0$) en las pérdidas del discriminador.

##### 2.2.4. Training objective
* **Problemas atacados**: Formulación matemática de la optimización del sistema completo.
* **Limitaciones de ese entonces**: Modelos propensos al colapso si la normalización se aplica sin control al entrenamiento de los discriminadores.
* **Soluciones alcanzadas**: Se definen dos pérdidas alternadas: $L_{pred1}$ para optimizar los discriminadores $D_f$ y $D_p$, y $L_{pred2}$ para entrenar el extractor y el detector, ponderados por un coeficiente $\alpha$ para regular la fuerza de la alineación de predicción.

### 3. Experiment
Presenta el diseño experimental, datasets de origen y destino, y los parámetros de entrenamiento.
* **Problemas atacados**: Evaluación empírica de la adaptación de dominio de vehículos en teledetección.
* **Limitaciones de ese entonces**: Dificultad para transferir detectores entrenados en datasets estadounidenses (e.g. COWC) hacia áreas con morfologías viales muy diferentes (e.g. Tokio, Japón).
* **Soluciones alcanzadas**: Diseño de pruebas cruzadas COWC (origen) $\to$ Tokio (destino).

#### 3.1. Dataset
* **Problemas atacados**: Estandarización de escalas y resoluciones óptimas.
* **Limitaciones de ese entonces**: Diferencias de altura de vuelo y ángulo de satélite distorsionan la escala física de los vehículos (píxel/metro).
* **Soluciones alcanzadas**: Ambas imágenes se re-muestrean a una resolución física común de 0.3m/píxel y se recortan en parches de 300x300, aplicando aumento de datos por rotación.

#### 3.2. Experimental Setting
Detalla el hardware, optimización y variantes de prueba.
* **Problemas atacados**: Comparación justa entre diferentes configuraciones de normalización adversaria.
* **Limitaciones de ese entonces**: Ausencia de análisis sobre si la normalización de clase (CWN) debía aplicarse al discriminador, al detector, o a ambos.
* **Soluciones alcanzadas**: 
  * **3.2.1. Implementation**: Uso de SSD con VGG-16 modificando los tamaños de default boxes para adaptarlos a vehículos.
  * **3.2.2. Training configuration**: Entrenamiento del detector base en COWC y afine por DA en Tokio, comparando: Plain adversarial (solo alineación de características), "w/o norm" (alineación de predicciones sin normalizar), "norm D and P" (CWN en discriminadores y detector) y "norm P" (CWN solo en detector y extractor de características).

### 4. Result and Discussion
Muestra los resultados de AP en el dataset de prueba.
* **Problemas atacados**: Evaluación cuantitativa y diagnóstico de fallas de convergencia.
* **Limitaciones de ese entonces**: Aplicar la normalización CWN a los discriminadores desestabiliza su aprendizaje, induciendo model collapse a menos que el factor $\alpha$ se reduzca drásticamente a 0.1.
* **Soluciones alcanzadas**:
  * La alineación de predicciones sin normalizar (w/o norm) ofrece mejoras muy pequeñas frente a la alineación de características.
  * La variante "norm P" (CWN aplicada solo al extractor y detector) evita el model collapse, logrando un AP de 78.7% (ganancia de 4% sobre la alineación básica adversaria de 74.7% y de 12.5% sobre el modelo sin adaptación de dominio de 66.2%).

### 5. Conclusion
Resume los resultados y propone líneas de investigación.
* **Problemas atacados**: Generalización del framework.
* **Limitaciones de ese entonces**: El ajuste de hiperparámetros de CWN fue empírico y limitado a una sola clase de vehículo en una arquitectura específica (SSD).
* **Soluciones alcanzadas**: Confirmar que la alineación adversaria en la salida con normalización selectiva de clase (CWN) en el generador de características es un método sumamente eficaz, proyectando su extensión a detectores generales multiclase y problemas de súper-resolución acoplada.
