# Artificial and beneficial – Exploiting artificial images for aerial vehicle detection

- **Key**: Weber2021ArtificialImagesVehicleDetection
- **Year**: 2021
- **Venue**: ISPRS Journal of Photogrammetry and Remote Sensing

## Resumen
Este artículo aborda el problema de la escasez de datos etiquetados para la detección de vehículos en imágenes aéreas (como maquinaria agrícola o de construcción) mediante un enfoque generativo simple pero efectivo. Los autores proponen un generador de imágenes aéreas cenitales que utiliza planos de diseño asistido por computadora (CAD 2D) simplificados y coloreados de forma aleatoria, los cuales se superponen sobre fondos artificiales con patrones de ruido fino y grueso. Para la detección, adaptan la arquitectura de RetinaNet eliminando los niveles superiores de la pirámide de características FPN ($P_5$-$P_7$) e incorporando un nivel inferior fino ($P_2$) para capturar objetos pequeños. También integran mecanismos de centerness de ATSS y una rama adicional de segmentación semántica para analizar cualitativamente la representación de características. Los experimentos basados en el dataset ISPRS Potsdam demuestran que las redes entrenadas únicamente con datos artificiales son capaces de generalizar y detectar vehículos reales (0.68 AP). Al combinar datos, la adición de 1000 imágenes artificiales a pequeños conjuntos reales (ej. 8 imágenes reales) eleva el AP desde niveles casi nulos hasta 0.736 AP, reduciendo a la mitad la cantidad de imágenes reales necesarias para alcanzar niveles óptimos de precisión. Además, analizan en profundidad la importancia de la composición de la imagen (foreground vs. background) y revelan que la colocación semántica (ej. evitar coches sobre tejados) influye de manera secundaria en el rendimiento del detector.

## Secciones y Subsecciones

### 1. Introducción
Presenta el contexto de la detección de vehículos en imágenes cenitales y la problemática de la escasez de datos etiquetados en aplicaciones del mundo real, especialmente en distribuciones desbalanceadas de cola larga (long-tail) para vehículos raros o especializados.
* **Problemas atacados**: Alto coste de etiquetado manual y la escasez física de imágenes de vehículos de baja frecuencia (ej. excavadoras, tractores) en comparación con coches de pasajeros.
* **Limitaciones de ese entonces**: Los detectores de aprendizaje profundo requieren miles de muestras para aprender invariancias ante variaciones de color de los vehículos, sombras, iluminación y texturas de fondo.
* **Soluciones alcanzadas**: Desarrollo de un generador de imágenes artificiales 2D de bajo coste computacional y diseño de experimentos sistemáticos con RetinaNet modificado para cuantificar el impacto del tamaño del dataset real y la simulación de clases raras.

### 2. Trabajo Relacionado
Clasifica y compara las estrategias existentes para mitigar la falta de datos: aumento de imágenes (Simple, Mixup, RICAP), aprendizaje por transferencia, adaptación de dominio (CORAL, CycleGAN), aprendizaje semi-supervisado (Noisy Student) y síntesis de imágenes (motores 3D, videojuegos, GANs).

* **Problemas atacados**: Identificación de los pros y contras de cada método de enriquecimiento de datos.
* **Limitaciones de ese entonces**: Los motores de renderizado 3D y videojuegos requieren el modelado manual costoso de texturas y entornos en 3D. Los enfoques semi-supervisados o GANs siguen necesitando una cantidad base considerable de muestras de la misma distribución objetivo.
* **Soluciones alcanzadas**: Posicionar la síntesis basada en CAD 2D como una alternativa barata y rápida que permite automatizar las anotaciones de caja (OBB y HBB) y de píxel de forma gratuita, sirviendo de base para aplicar técnicas posteriores de regularización o aumento.

### 3. Framework
Describe la estructura integrada del sistema: el detector de objetos modificado y el generador generativo de imágenes.

* **Problemas atacados**: Ajustar detectores genéricos de imágenes naturales a las restricciones de escala aérea y formular el proceso de dibujo vectorial del coche.
* **Limitaciones de ese entonces**: Redes estándar como RetinaNet para COCO están optimizadas para objetos que ocupan gran parte del encuadre y no para vehículos pequeños de pocos píxeles observados desde arriba.
* **Soluciones alcanzadas**: Rediseño del flujo FPN y simplificación geométrica de los blueprints vehiculares para crear parches sintéticos.

#### 3.1 Detector de Objetos
Detalla los cambios a RetinaNet: sustitución de los niveles FPN gruesos por un nivel P2 de alta resolución espacial, adición de escalado por nivel de FCOS y el predictor de centerness de ATSS. También introduce una rama complementaria de segmentación.
* **Problemas atacados**: Baja resolución de las características en mapas de grano grueso y falsos positivos por anclas mal alineadas con el centro físico del objeto.
* **Limitaciones de ese entonces**: Las capas altas FPN ($P_5$-$P_7$) tienen campos receptivos demasiado grandes (ej. 10x10 píxeles de activación) que no se activan con vehículos pequeños.
* **Soluciones alcanzadas**: Eliminación de las capas predictivas $P_5$-$P_7$, derivación de predicciones a partir de $P_2$ y $P_3$ unicamente. Se implementa la rama de segmentación semántica con pérdida combinada Dice y Binary Cross-Entropy para estudiar cualitativamente la respuesta de los píxeles.

#### 3.2 Generador de Imágenes Artificiales
Explica cómo se preparan los planos CAD 2D (blueprints), el flujo de coloreado por tipo de superficie, deformación del coche y la inyección de ruido grueso y fino en el fondo.
* **Problemas atacados**: Variabilidad geométrica e intra-clase y el modelado de fondos que prevengan falsos positivos.
* **Limitaciones de ese entonces**: Usar representaciones CAD crudas incluye marcas de acotación de ingeniería inútiles. Fondos artificiales planos provocan que el modelo aprenda a discriminar solo la ausencia de textura.
* **Soluciones alcanzadas**: Proceso de limpieza de planos CAD 2D para aislar la vista superior del vehículo (Figura 3). El generador crea un fondo gris (promedio de ImageNet) y le superpone ruido grueso (interpolación bicúbica sobre una cuadrícula $10 \times 10$) y ruido fino gaussiano, imitando pavimentos reales. Los vehículos se colorean, deforman hasta $\pm 5\%$ y se recortan opcionalmente para simular oclusiones.

### 4. Datos
Introduce el conjunto de datos de Potsdam (ISPRS 2D Semantic Labeling) y explica la conversión de sus máscaras de segmentación en cajas delimitadoras HBB y OBB (Figura 6).
* **Problemas atacados**: Obtención de una base de datos real consistente para realizar simulaciones de escasez de datos.
* **Limitaciones de ese entonces**: Potsdam provee máscaras de píxeles para segmentación semántica urbana, pero no anotaciones nativas de cajas para detección.
* **Soluciones alcanzadas**: Extracción morfológica de los contornos de la clase "coche" y estimación geométrica automática de las cajas alineadas con los objetos, depurando errores como oclusiones extremas y tranvías mal etiquetados.

### 5. Experimentos
Presenta los resultados de los experimentos unificados entrenados con PyTorch-Lightning en GPUs Tesla V100.

* **Problemas atacados**: Medición del beneficio de los datos sintéticos combinados y análisis de componentes del generador.
* **Limitaciones de ese entonces**: Evaluar la síntesis de datos sin medir el impacto del GSD o de la capacidad del extractor (backbone).
* **Soluciones alcanzadas**: Demostración de la robustez del método variando sistemáticamente el número de muestras reales ($N_r$), las técnicas de aumento y la composición del fondo.

#### 5.1 Configuración Experimental
Describe los parches de 600x600 px submuestreados a 300x300 px (GSD resultante de 0.10 m/px) y el uso de la política de entrenamiento "one-cycle".
* **Problemas atacados**: Eficiencia temporal y convergencia estable del optimizador en datasets de tamaño variable.
* **Limitaciones de ese entonces**: Épocas variables dificultan comparar pérdidas entre datasets de diferente escala.
* **Soluciones alcanzadas**: Fijar el entrenamiento a un número constante de 2500 iteraciones en lugar de épocas fijas, usando una tasa de aprendizaje con rampas de ascenso y descenso controlado (Smith, 2018).

#### 5.2 Combinación de Imágenes Reales y Artificiales
Compara los grupos de entrenamiento "baseline" (solo real, $N_r = 8$ a $2039$) contra "combination" (real + 1000 artificiales, $N_a=1000$).
* **Problemas atacados**: Cuantificar el salto de AP en escenarios de escasez extrema.
* **Limitaciones de ese entonces**: Se desconocía el punto de saturación donde los datos sintéticos dejan de aportar valor.
* **Soluciones alcanzadas**: Si se disponen de solo 8 imágenes reales, añadir 1000 imágenes sintéticas sube el rendimiento de 0.05 a 0.736 AP. La ganancia se mantiene significativa por debajo de 100 imágenes reales y converge a 0.95 AP en conjuntos grandes (Figura 7). Además, se comprueba que el beneficio es independiente de la capacidad de la red (de ResNet-18 a ResNeXt-50, ver Tabla 1) y del GSD de entrada (Tabla 2).

#### 5.3 Detalles del Generador de Imágenes Artificiales
Mide de forma aislada el aporte de cada módulo del generador entrenando únicamente con datos sintéticos ($N_r = 0$).
* **Problemas atacados**: Aporte marginal de las líneas de contorno, las deformaciones y el ruido de fondo.
* **Limitaciones de ese entonces**: Modelar la geometría del coche con formas de un solo color plano reducía el AP notablemente.
* **Soluciones alcanzadas**: 1) Añadir contornos negros al coche aporta +0.032 AP. 2) Simular oclusiones cortando vehículos aporta +0.028 AP. 3) Introducir deformaciones geométricas aporta +0.046 AP. 4) Añadir ruido de fondo (grueso y fino) aporta +0.06 AP, demostrando ser el factor clave para mitigar falsos positivos en pavimentos reales.

#### 5.4 La Importancia de la Composición de la Imagen
Analiza cuantitativamente la interacción del objeto y el fondo cruzando componentes artificiales y reales (Tabla 4).
* **Problemas atacados**: Determinar si el realismo visual del fondo es suficiente por sí solo o si influyen los factores de colocación semántica.
* **Limitaciones de ese entonces**: Se teorizaba que colocar coches sobre tejados o copas de árboles (violando las restricciones de circulación reales) destruiría la capacidad de generalización del detector.
* **Soluciones alcanzadas**: 1) La diferencia de rendimiento entre coches reales y artificiales sobre el mismo fondo sintético es mínima (<0.05 AP), lo que valida el realismo de las características aprendidas con planos CAD 2D. 2) Colocar vehículos sobre fondos reales sin lógica espacial (ej. coches flotantes o sobre edificios) apenas reduce el mAP de 0.93 a 0.89 AP, confirmando que el contexto semántico de la escena es un factor secundario para el aprendizaje del detector de vehículos.

### 6. Conclusión y Trabajo Futuro
* **Problemas atacados**: Resumen de descubrimientos sobre la interacción de fondos y objetos sintéticos.
* **Limitaciones de ese entonces**: Queda una brecha de rendimiento (gap) por cerrar entre modelos entrenados solo con datos sintéticos (0.68 AP) y datos reales (0.94 AP).
* **Soluciones alcanzadas**: Demostración de que el fondo real actúa principalmente como un regularizador contra falsos positivos. Se propone como trabajo futuro investigar técnicas de armonización y adaptación de dominio de las OBBs sintetizadas, así como analizar qué detalles visuales específicos del vehículo inducen las mayores ganancias de aprendizaje.
