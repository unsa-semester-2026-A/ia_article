# Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection

- **Key**: Akyon2022SAHI
- **Year**: 2022
- **Venue**: ICIP

## Resumen
SAHI (Slicing Aided Hyper Inference) es un framework y pipeline de código abierto diseñado para optimizar la detección de objetos pequeños y distantes en imágenes de muy alta resolución (por ejemplo, imágenes aéreas capturadas por drones o satélites). Los detectores estándar suelen fallar en este tipo de objetos debido a que están representados por muy pocos píxeles y carecen de suficiente detalle visual. SAHI introduce una metodología genérica que divide las imágenes en porciones o rodajas (slicing) solapadas tanto durante la inferencia como en el ajuste fino (fine-tuning). Al redimensionar estas porciones, los objetos pequeños aumentan su área de píxeles relativa dentro de la red, facilitando su detección. Esta aproximación es agnóstica al modelo y se puede aplicar directamente sobre cualquier detector existente (ej. YOLO, Detectron2, MMDetection) sin alterar su arquitectura interna. Los resultados en VisDrone y xView demuestran mejoras masivas de hasta +14.5% mAP, permitiendo procesar imágenes de alta resolución en hardware con memoria limitada.

## Secciones y Subsecciones

### 1. Introduction
Presenta el reto que supone la detección de objetos a escalas reducidas y contextualiza la discrepancia entre datasets de laboratorio y aplicaciones del mundo real.
* **Problemas atacados**: El pobre rendimiento de los detectores de objetos modernos al procesar imágenes de alta resolución (como 4K o tomas de satélite) que contienen objetos pequeños y lejanos que cubren áreas inferiores al 1% de la imagen.
* **Limitaciones de ese entonces**: Los detectores se entrenan y evalúan principalmente en datasets como COCO o Pascal VOC, cuyas imágenes son de baja resolución (640x480) y contienen objetos grandes que cubren en promedio el 60% de la altura de la imagen. Al aplicarlos en imágenes aéreas de drones o satélites con objetos de pocos píxeles, la precisión decae drásticamente debido a la falta de resolución y detalles contextuales de los objetos pequeños.
* **Soluciones alcanzadas**: Se propuso SAHI, un pipeline que realiza slicing (división en rodajas) de la imagen de prueba para generar parches de menor tamaño que son posteriormente redimensionados antes de entrar a la red. Esto aumenta la resolución efectiva del objeto pequeño, resolviendo además el problema de memoria de GPU que se asocia a procesar imágenes de muy alta resolución completas.

### 2. Related Work
Revisa los detectores de objetos modernos y las metodologías especializadas en la detección de objetos de baja escala.
* **Problemas atacados**: La ineficiencia y complejidad de adaptar arquitecturas generales para la detección de objetos a escalas muy reducidas sin incurrir en costos de entrenamiento prohibitivos.
* **Limitaciones de ese entonces**: Métodos existentes para objetos pequeños requieren rediseñar el detector (como añadir super-resolución JCS-Net, visual attention en STDnet o redes generativas GAN para caras), lo que obliga a entrenar desde cero y dificulta reutilizar pesos preentrenados en grandes datasets como COCO. Otras técnicas como aumento de datos mediante copia de objetos requieren máscaras de segmentación detalladas que no siempre están disponibles en datasets de detección estándar.
* **Soluciones alcanzadas**: Se diseñó un framework genérico basado en slicing externo que no requiere modificar la arquitectura del detector ni entrenarlo desde cero. Es compatible con detectores de una y dos etapas, así como con detectores libres de anclas recientes (FCOS, VarifocalNet, TOOD), y permite usar pesos preentrenados directamente.

### 3. Proposed Approach
Explica los dos componentes principales del framework SAHI: ajuste fino asistido por slicing e hiper-inferencia asistida por slicing.
* **Problemas atacados**: La desalineación de escalas entre el entrenamiento de baja resolución y la inferencia de alta resolución, y la pérdida de objetos de gran escala al fragmentar la imagen.
* **Limitaciones de ese entonces**: El procesamiento ingenuo por parches puede fragmentar objetos de gran tamaño ubicados en los bordes de los parches, impidiendo su correcta detección. Además, entrenar con imágenes completas de alta resolución es inviable por memoria de GPU.
* **Soluciones alcanzadas**: Se propuso un pipeline dual que incluye: 1) Ajuste fino asistido por slicing (SF) para robustecer la red a parches; 2) Hiper-inferencia asistida por slicing (SAHI) para la fase de test, que puede opcionalmente combinarse con inferencia de la imagen completa (FI) y fusiona todas las predicciones usando supresión no máxima (NMS).

#### Slicing Aided Fine-tuning (SF)
Describe el proceso de preparación de datos de entrenamiento mediante parches.
* **Problemas atacados**: La falta de familiaridad de las redes preentrenadas con las escalas y proporciones de objetos de drones o satélites.
* **Limitaciones de ese entonces**: Ajustar directamente un modelo preentrenado en COCO sobre imágenes completas de drones no corrige el problema de la falta de píxeles en los objetos pequeños debido al downsampling de las capas convolucionales profundas.
* **Soluciones alcanzadas**: Se propuso extraer parches solapados de dimensiones $M \times N$ de las imágenes de entrenamiento. Estos parches se redimensionan a escalas más grandes (e.g. entre 800 y 1333 píxeles) antes de la retropropagación, incrementando el tamaño relativo del objeto pequeño. Estos parches se mezclan con las imágenes completas originales para evitar que el detector pierda la capacidad de detectar objetos de gran tamaño.

#### Slicing Aided Hyper Inference (SAHI)
Detalla el flujo de trabajo de la inferencia dividida y la combinación de predicciones.
* **Problemas atacados**: La detección en tiempo de inferencia de objetos densos y pequeños en imágenes de muy alta resolución sin agotar la memoria VRAM de la GPU.
* **Limitaciones de ese entonces**: Alimentar imágenes de tamaño superior a 4000x4000 píxeles a detectores modernos causa fallos por memoria insuficiente (Out of Memory - OOM) en GPUs convencionales.
* **Soluciones alcanzadas**: Durante la inferencia, la imagen de entrada se corta en $l$ parches de tamaño $M \times N$ con solapamiento configurable. Cada parche se redimensiona y se procesa por separado. Opcionalmente, se corre una inferencia completa de la imagen (Full Inference - FI). Todas las cajas predichas se reproyectan a las coordenadas originales de la imagen y se consolidan mediante NMS con un umbral de coincidencia $T_m$ y de confianza $T_d$.

### 4. Results
Presenta las pruebas experimentales y análisis cualitativos/cuantitativos de SAHI.
* **Problemas atacados**: Evaluar la generalidad y la ganancia en precisión del método SAHI sobre múltiples detectores y conjuntos de datos reales de drones y satélites.
* **Limitaciones de ese entonces**: Es común que las propuestas de detección por parches solo se evalúen con un único detector cerrado, dificultando evaluar su impacto agnóstico.
* **Soluciones alcanzadas**: Se implementó SAHI sobre FCOS, VarifocalNet (VFNet) y TOOD usando MMDetection. Las pruebas en VisDrone y xView demostraron incrementos drásticos en AP50 para todos los modelos.

#### Datasets
Detalla los conjuntos de datos de drones y satélites utilizados.
* **Problemas atacados**: Validar la efectividad en condiciones de oclusión severa, cambios de iluminación (VisDrone) y resolución extrema con clases altamente desbalanceadas (xView).
* **Limitaciones de ese entonces**: La mayoría de las evaluaciones clásicas se limitan a datasets estándar con objetos de tamaño medio y distribución uniforme.
* **Soluciones alcanzadas**: Se empleó VisDrone (8599 imágenes con 10 clases aéreas) y xView (más de 1 millón de instancias satelitales en 60 clases), ambos caracterizados por objetos con anchos inferiores al 1% del ancho de la imagen.

#### Setup y Resultados
Describe los parámetros de entrenamiento, inferencia y los resultados de precisión por tamaño de objeto.
* **Problemas atacados**: La optimización de hiperparámetros del slicing (solapamiento, dimensiones del parche) y la consolidación de cajas redundantes.
* **Limitaciones de ese entonces**: La división en parches genera predicciones redundantes e incrementa falsos positivos en los límites de los parches cuando no hay solapamiento adecuado o NMS está mal configurado.
* **Soluciones alcanzadas**: 1) Se determinó que SAHI solo (sin fine-tuning) aumenta el AP50 en +6.8% (FCOS), +5.1% (VFNet) y +5.3% (TOOD); 2) Combinar SAHI con fine-tuning asistido por slicing (SF) produce ganancias acumuladas de +12.7%, +13.4% y +14.5% respectivamente; 3) Un solapamiento de parches del 25% (PO) mejora la detección en bordes, aunque incrementa ligeramente los falsos positivos en objetos grandes; 4) En xView (cuyo baseline es casi cero de ~2% AP), SF+SAHI elevó el AP50 hasta un 20.6% usando TOOD.

### 5. Conclusion
Resume los aportes y alcances del framework SAHI.
* **Problemas atacados**: El compromiso de diseño entre precisión de detección, costo computacional y presupuesto de memoria del hardware.
* **Limitaciones de ese entonces**: Procesar imágenes de alta resolución mediante fuerza bruta exige hardware industrial costoso y no es escalable para aplicaciones en sistemas embebidos de drones.
* **Soluciones alcanzadas**: SAHI ofrece un balance práctico: incrementa el tiempo de cómputo de manera lineal respecto al número de parches, pero mantiene el consumo de memoria de GPU fijo y bajo. El tamaño del parche puede calibrarse dinámicamente para ajustarse al presupuesto de hardware del cliente, ofreciendo un pipeline sumamente versátil y listo para producción.
