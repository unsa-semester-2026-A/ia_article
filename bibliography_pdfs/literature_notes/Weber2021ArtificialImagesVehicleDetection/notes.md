# Artificial and beneficial – Exploiting artificial images for aerial vehicle detection

- **Key**: Weber2021ArtificialImagesVehicleDetection
- **Year**: 2021
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing

## Resumen
Este artículo presenta una metodología generativa simple y altamente efectiva para mitigar la escasez de datos anotados en la detección de vehículos aéreos (especialmente para clases de larga cola como maquinaria agrícola o de construcción). En lugar de recurrir a costosos motores de renderizado 3D o simulaciones fotorrealistas complejas, los autores proponen un generador de imágenes 2D que toma planos CAD bidimensionales simplificados (blueprints) de vehículos, los pinta y deforma de forma aleatoria, y los superpone en fondos neutros sintetizados con ruido Gaussiano o en fondos reales de percepción remota. Utilizando un detector RetinaNet modificado (con un pirámide de características truncado optimizado para objetos pequeños, rama de centerness y un cabezal de segmentación semántica auxiliar), los experimentos demuestran que agregar imágenes artificiales a conjuntos pequeños de datos reales mejora drásticamente la precisión. En escenarios de datos reales extremadamente limitados (p. ej., solo 8 imágenes), la adición de 1000 imágenes artificiales eleva el AP medio (mAP) en hasta 70 puntos. El estudio analiza además el impacto crítico de la composición de la imagen, revelando que el ruido y la estructura del fondo son más determinantes que el fotorrealismo de los vehículos para reducir la tasa de falsos positivos en objetos similares (hard negatives).

## Secciones y Subsecciones

### 1. Introduction
Se expone el problema de la escasez de datos etiquetados en tomas aéreas y su impacto en la detección con aprendizaje profundo. A pesar del éxito de detectores basados en CNNs, la recolección y anotación manual es sumamente costosa, especialmente para clases con distribuciones desbalanceadas (long-tail) donde automóviles comunes abundan pero vehículos industriales o agrícolas son muy raros.
* **Problemas atacados**: La gran demanda de datos anotados de las redes profundas frente a la escasez de muestras reales para clases vehiculares raras y la enorme variación intra-clase (colores, formas, entornos).
* **Limitaciones de ese entonces**: Las soluciones tradicionales implican campañas costosas de vuelo o el modelado manual en 3D de alta complejidad para motores gráficos, lo cual sigue requiriendo un esfuerzo humano sustancial.
* **Soluciones alcanzadas**: Desarrollo de un generador de imágenes artificiales aéreas basado en planos 2D CAD simplificados superpuestos en fondos sintéticos texturizados. Se simula la escasez de datos entrenando modelos con la clase común "autos" del dataset Potsdam a escalas reducidas de muestras reales.

### 2. Related work
Revisión de la literatura sobre metodologías para abordar la escasez de datos. Se subdivide en aumentos de imagen, aprendizaje por transferencia, adaptación de dominio, aprendizaje semi-supervisado, generación de imágenes sintéticas 3D/videojuegos, y redes generativas adversarias (GANs).
* **Problemas atacados**: Comparar y justificar el uso de un generador sintético geométrico frente a técnicas de aumento tradicionales o aprendizaje no supervisado.
* **Limitaciones de ese entonces**: Técnicas de aumento clásico (cropping, flipping) no agregan información fuera de la distribución original. La adaptación de dominio y el aprendizaje semi-supervisado (como Noisy Student o CycleGAN) requieren conjuntos de datos de origen ya anotados y extensos. La simulación 3D requiere texturas detalladas y trabajo de animación complejo.
* **Soluciones alcanzadas**: Justificación de que la generación artificial 2D es la opción más flexible y económica para clases raras, sirviendo como base sobre la cual pueden aplicarse técnicas complementarias de aumentos, armonización o CycleGANs.

### 3. Framework
Descripción de los dos componentes principales del sistema propuesto: el detector de objetos optimizado y el generador de imágenes artificiales 2D.
* **Problemas atacados**: Adaptación del detector de objetos terrestre a tomas aéreas de resolución sub-métrica y diseño de un pipeline de generación que minimice la complejidad visual sin comprometer la transferencia de características.
* **Limitaciones de ese entonces**: Los detectores comunes como RetinaNet están optimizados para MS COCO, el cual contiene objetos grandes y vistas a nivel del suelo, resultando en un rendimiento deficiente cuando se procesan objetos aéreos pequeños de pocos píxeles.
* **Soluciones alcanzadas**: Adaptación de RetinaNet a parches fijos pequeños y desarrollo de un generador que automatiza la creación de máscaras y anotaciones de OBB/HBB.

#### 3.1. Object detector
Modificaciones específicas introducidas sobre la arquitectura de RetinaNet y su extractor de características ResNet-50.
* **Problemas atacados**: El desperdicio computacional de evaluar características gruesas y el desalineamiento entre el centro de los anclajes y los objetos pequeños en vistas aéreas.
* **Limitaciones de ese entonces**: Las capas superiores de la pirámide de características FPN (P5 a P7) tienen campos receptivos demasiado grandes para objetos aéreos pequeños, lo que produce falsas alarmas. La regresión estándar Smooth L1 no penaliza adecuadamente desalineamientos finos del centro.
* **Soluciones alcanzadas**: Truncamiento de la pirámide eliminando P5-P7 e incorporando la capa de alta resolución P2. Se agregaron una rama predictora de "centerness" (para ponderar la confianza del anclaje), una rama auxiliar de segmentación semántica en P2 y el uso de factores de escala ajustables.

#### 3.2. Artificial Image Generator
Pasos del algoritmo de generación sintética 2D (Fig. 5).
* **Problemas atacados**: Variabilidad geométrica y de color de los objetos sintetizados y simulación de oclusiones sin usar gráficos 3D.
* **Limitaciones de ese entonces**: Los objetos 2D puros superpuestos de forma estática carecen de variabilidad de forma y detalles de contorno, pareciendo artificiales a las capas iniciales de la CNN.
* **Soluciones alcanzadas**: Uso de 2D blueprints CAD vectoriales pintados de forma aleatoria, aplicación de deformaciones de escala ($\pm 5\%$), rotaciones de $360^\circ$, cortes geométricos aleatorios para simular oclusiones parciales e introducción de ruido Gaussiano de frecuencia fina y gruesa (interpolación bicúbica) en el fondo.

### 4. Data
Manejo y adaptación del conjunto de datos real ISPRS Potsdam para las pruebas de detección.
* **Problemas atacados**: Adaptar un dataset diseñado para segmentación semántica de píxeles a tareas de regresión de cajas delimitadoras.
* **Limitaciones de ese entonces**: Potsdam contiene etiquetas de píxeles continuas pero carece de anotaciones de cajas orientadas individuales (OBB) necesarias para detectores modernos. Además, sufre de oclusiones no etiquetadas bajo árboles sin hojas en otoño/invierno.
* **Soluciones alcanzadas**: Procesamiento de contornos de la clase "car" mediante técnicas de visión por computadora para derivar cajas orientadas (OBB) y alineadas con los ejes (HBB) (Fig. 6), eliminando las instancias parcialmente truncadas de menos de 20 píxeles.

### 5. Experiments
Protocolo de experimentación, métricas y resultados empíricos divididos en cuatro sub-experimentos.
* **Problemas atacados**: Evaluar la generalización de la red entrenada con imágenes sintéticas al dominio de imágenes reales bajo variaciones de GSD, capacidad del backbone y técnicas de aumento.
* **Limitaciones de ese entonces**: Falta de análisis riguroso sobre cómo influye la cantidad de muestras reales en el beneficio neto aportado por el dataset sintético.
* **Soluciones alcanzadas**: Configuración de pruebas usando PyTorch Lightning sobre GPUs V100, evaluando mediante métricas MS COCO AP al umbral IoU=0.5 con 8 repeticiones aleatorias por experimento.

#### 5.1. Experimental Setup
Definición de las particiones de datos y el escalado espacial (GSD).
* **Problemas atacados**: Ajustar el campo de visión de las imágenes aéreas Potsdam gigantescas a la capacidad del detector.
* **Limitaciones de ese entonces**: Procesar imágenes de $6000 \times 6000$ píxeles directamente satura la memoria GPU.
* **Soluciones alcanzadas**: Segmentación en parches de $600 \times 600$ píxeles y reducción de escala (down-sampling) a $300 \times 300$ para duplicar el GSD original a 0.10 m/px, acelerando drásticamente el entrenamiento con una pérdida de precisión marginal.

#### 5.2. Combining real and artificial images
Resultados cuantitativos y cualitativos de la fusión de datasets reales y sintéticos.
* **Problemas atacados**: Cuantificar el beneficio de las imágenes artificiales a medida que aumenta el volumen de datos reales de entrenamiento.
* **Limitaciones de ese entonces**: La suposición de que los datos sintéticos simples degradan el rendimiento semántico de la red si se dispone de muestras reales.
* **Soluciones alcanzadas**: La adición de 1000 imágenes artificiales estabiliza las predicciones del detector y eleva el AP de forma masiva en el rango de menos de 100 imágenes reales (p. ej., con 8 imágenes reales el AP sube de un valor inservible a 0.70). Con datasets reales grandes (>200 imágenes), las curvas convergen al mismo límite de 0.95 AP, demostrando que los datos artificiales no dañan el rendimiento general.

##### Impact of network capacity
* **Problemas atacados**: Validar si el beneficio de las imágenes artificiales depende del tamaño del backbone.
* **Limitaciones de ese entonces**: Redes con mayor capacidad podrían sobreajustar (overfit) a las características simples de los planos CAD.
* **Soluciones alcanzadas**: Pruebas con ResNet-18, 50, 152, ResNeXt y WideResNet demuestran que el beneficio es consistente en todos los backbones, determinando que ResNet-50 es el balance óptimo.

##### Impact of ground sampling distance
* **Problemas atacados**: Evaluar el impacto de la resolución espacial (GSD) en la transferencia de datos sintéticos.
* **Limitaciones de ese entonces**: Falta de análisis de la degradación del rendimiento al perder detalles sub-métricos.
* **Soluciones alcanzadas**: Se demuestra que el beneficio sintético se mantiene en GSDs de 0.05, 0.10, 0.15 y 0.20 m/px, con una reducción predecible de la precisión general a resoluciones más gruesas por la pérdida de detalles finos (Table 2).

##### Comparison with image augmentation
* **Problemas atacados**: Determinar la relación de complementariedad entre aumentos de datos y la síntesis artificial.
* **Limitaciones de ese entonces**: La creencia de que aumentos avanzados (como expandir y recortar de SSD) eliminan la necesidad de sintetizar muestras.
* **Soluciones alcanzadas**: Se demuestra que el aumento radiométrico y geométrico por rotación/flipping es complementario a la síntesis. Entrenar con imágenes combinadas y todos los aumentos activos proporciona la curva de AP más alta y estable (Fig. 11).

#### 5.3. Details of the artificial image generator
Ablación sistemática de los componentes del generador sintético.
* **Problemas atacados**: Identificar qué componentes geométricos o visuales del generador 2D aportan mayor valor al entrenamiento del modelo.
* **Limitaciones de ese entonces**: Falta de guías de diseño que especifiquen si el contorno, el color o el fondo de los objetos sintetizados son prioritarios.
* **Soluciones alcanzadas**: El desglose en la Table 3 revela que: 1) eliminar el contorno negro de los vehículos reduce el AP en 0.032, 2) omitir oclusiones parciales disminuye el AP en 0.028, y 3) la adición de ruido Gaussiano fino y grueso en el fondo aporta la mayor ganancia individual (+0.06 AP), ya que entrena a la red para diferenciar la estructura de los vehículos del ruido del terreno.

#### 5.4. The importance of image composition
Análisis cualitativo y cuantitativo del impacto del fondo y la semántica en la generalización de la red.
* **Problemas atacados**: Evaluar cómo influyen las vecindades espaciales y los contrastes de los objetos en la aparición de falsos positivos (hard negatives).
* **Limitaciones de ese entonces**: La creencia de que el rendimiento deficiente de modelos sintéticos se debe únicamente al modelado simplificado de los objetos.
* **Soluciones alcanzadas**: Se identifican y documentan dos aspectos cruciales detallados a continuación.

##### The effect of background
* **Problemas atacados**: Analizar visualmente la aparición de falsas alarmas en el terreno.
* **Limitaciones de ese entonces**: Los mapas de activación semántica de modelos entrenados en fondos lisos muestran un alto nivel de ruido en bordes estructurados reales.
* **Soluciones alcanzadas**: El análisis de activación semántica (Fig. 12) demuestra que entrenar con fondos lisos fuerza a la red a clasificar cualquier estructura física como vehículo. Agregar ruido al fondo sintético o incorporar solo 8 imágenes reales educa a la red para suprimir las activaciones en el terreno de fondo.

##### Image composition
* **Problemas atacados**: Evaluar la necesidad de ubicar los objetos en posiciones semánticamente coherentes (p. ej. autos solo sobre carreteras).
* **Limitaciones de ese entonces**: La suposición de que los vehículos colocados de forma incongruente (sobre techos o árboles) degradan severamente el aprendizaje de características de contexto.
* **Soluciones alcanzadas**: Experimentos con cuatro combinaciones cruzadas de vehículos y fondos reales/artificiales (Table 4) revelan que la colocación semánticamente incorrecta de vehículos sobre fondos aleatorios reales provoca una pérdida insignificante de AP (0.89 vs 0.93), demostrando que la red se enfoca prioritariamente en la geometría local del vehículo y que la coherencia de contexto espacial es secundaria.

### 6. Conclusion and future work
Resumen de las conclusiones del estudio y propuestas de desarrollo.
* **Problemas atacados**: Consolidar las pautas para el desarrollo de generadores de datos eficientes para visión aérea.
* **Limitaciones de ese entonces**: Campañas costosas de etiquetado real que inhiben el despliegue de soluciones deep learning en ODAI.
* **Soluciones alcanzadas**: Demostración de que planos CAD 2D simples y un modelado texturizado básico del fondo permiten entrenar detectores robustos y sugieren para el futuro el desarrollo de modelos que desacoplen y modelen de forma independiente los factores de variación de fondo y primer plano.
