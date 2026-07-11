# Adapting Vehicle Detector to Target Domain by Adversarial Prediction Alignment

- **Key**: Koga2021VehicleDomainAdaptation
- **Year**: 2021
- **Venue**: IGARSS

## Resumen
Este artículo propone una novedosa técnica de adaptación de dominio (Domain Adaptation - DA) no supervisada aplicada a la detección de vehículos en imágenes satelitales. En la práctica, la diferencia de distribución de datos entre el conjunto de entrenamiento (dominio fuente) y el de prueba (dominio objetivo) deteriora severamente el rendimiento de los detectores de objetos. Los métodos tradicionales de adaptación de dominio suelen centrarse exclusivamente en alinear el extractor de características, dejando el clasificador final sin adaptar. Este trabajo introduce la alineación en el espacio de salida de predicción (Adversarial Prediction Alignment), alineando simultáneamente tanto las predicciones de regresión de ubicación como las puntuaciones de confianza de clase mediante entrenamiento adversario. Para superar el desequilibrio de clases extremo en la detección de objetos (donde el fondo predomina sobre los vehículos), se propone la Normalización de Pesos de Clase (Class Weight Normalization - CWN). El método mejora en más de 5% el AP (Average Precision) al adaptar desde el dataset COWC a imágenes aéreas de Japón.

## Secciones y Subsecciones

### 1. Introduction
Presenta el desafío del cambio de dominio en detección de vehículos y las limitaciones de las estrategias de adaptación existentes que solo actúan sobre las características intermedias.
* **Problemas atacados**: La degradación severa del rendimiento de los detectores de vehículos cuando se aplican a imágenes satelitales obtenidas bajo diferentes condiciones o localizaciones geográficas que las de entrenamiento.
* **Limitaciones de ese entonces**: Los métodos de adaptación de dominio convencionales encuentran un espacio de características común (feature alignment), pero tras la alineación aún persisten sutiles diferencias que el clasificador e identificador final del detector (que fue entrenado puramente en la fuente) no puede resolver, deteriorando la precisión.
* **Soluciones alcanzadas**: Se propuso adaptar directamente el clasificador e identificador final al dominio objetivo alineando el espacio de salida de predicciones (tanto la confianza de clase como la regresión de coordenadas de localización) mediante entrenamiento adversario guiado por discriminadores.

### 2. Methodology
Explica la base conceptual y el pipeline del framework propuesto para transferir la confianza y estructura de las predicciones del dominio fuente al dominio objetivo.
* **Problemas atacados**: La adaptación de la cabeza de predicción estructurada (bounding boxes y confianzas de clase) de un detector de objetos a un dominio sin etiquetar.
* **Limitaciones de ese entonces**: Técnicas de adaptación en el clasificador como la minimización de entropía solo sirven para confianzas de clase y no se pueden aplicar directamente a la regresión de coordenadas en detección de objetos.
* **Soluciones alcanzadas**: Se diseñó un framework de alineación adversaria multirrama que procesa de manera conjunta e integrada el vector concatenado de regresión de localización y clasificación.

#### 2.1. Vehicle Detector
Detalla el detector base seleccionado para las pruebas.
* **Problemas atacados**: La necesidad de un detector de objetos rápido, simple y robusto adecuado para imágenes satelitales.
* **Limitaciones de ese entonces**: Detectores complejos de dos etapas introducen componentes intermedios adicionales que complican la propagación del gradiente adversario para la alineación.
* **Soluciones alcanzadas**: Se adoptó el detector SSD (Single Shot MultiBox Detector) con VGG-16, el cual procesa parches espaciales a través de regresores y clasificadores directos acoplados a cajas por defecto (default boxes).

#### 2.2. Prediction Alignment
Detalla los pasos de alineación a nivel de características y predicciones.
* **Problemas atacados**: La inestabilidad en el entrenamiento y el colapso del modelo cuando se intenta alinear únicamente las salidas finales de la red.
* **Limitaciones de ese entonces**: El entrenamiento adversario en el espacio de predicción de forma aislada oscila violentamente y no converge si las representaciones latentes intermedias difieren demasiado.
* **Soluciones alcanzadas**: Se combinó la alineación del espacio de predicción con una alineación de características previa en la red dorsal, ejecutando ambas mediante optimización adversaria alternada.

##### 2.2.1. Feature alignment
Describe la alineación a nivel de extractor de características.
* **Problemas atacados**: Lograr una convergencia estable en el entrenamiento adversario y reducir la discrepancia latente básica del dominio.
* **Limitaciones de ese entonces**: Alinear mapas de características globales en imágenes satelitales no considera el contexto local de los objetos pequeños.
* **Soluciones alcanzadas**: Se aplicó alineación adversaria a nivel de unidades de parches de $3 \times 3$ píxeles sobre el mapa de características más superficial (que procesa los vehículos), entrenando un discriminador dedicado $D_f$.

##### 2.2.2. Prediction alignment
Explica la alineación a nivel de salidas del detector.
* **Problemas atacados**: La desalineación fina de los vectores de predicción de cajas delimitadoras y etiquetas del detector entre dominios.
* **Limitaciones de ese entonces**: Carecer de alineación en las predicciones causa que el clasificador del dominio fuente prediga con baja confianza sobre el dominio objetivo.
* **Soluciones alcanzadas**: Se concatenaron los offsets de localización $(cx, cy, w, h)$ y las confianzas de clase de fondo y vehículo en un solo vector. Se entrenó un discriminador de predicciones $D_p$ para forzar al detector a producir distribuciones idénticas en ambos dominios.

##### 2.2.3. Class weight normalization
Aborda el desbalance extremo entre el fondo y la clase vehículo.
* **Problemas atacados**: La saturación del gradiente de optimización provocada por la inmensa cantidad de predicciones de fondo, que ahogan el aprendizaje de la clase minoritaria vehículo.
* **Limitaciones de ese entonces**: En imágenes satelitales, más del 99% de las cajas por defecto contienen fondo. La alineación adversaria naive se enfoca en alinear el fondo, ignorando a los vehículos reales.
* **Soluciones alcanzadas**: Se introdujo la técnica CWN que calcula dinámicamente en cada minibatch un peso inversamente proporcional a la cantidad de muestras predichas para cada clase y reescala los gradientes propagados hacia el detector y el extractor.

##### 2.2.4. Training objective
Establece la formulación matemática de la función de pérdida general.
* **Problemas atacados**: Balancear el entrenamiento del detector supervisado en la fuente con los objetivos adversarios en el objetivo.
* **Limitaciones de ese entonces**: Integrar múltiples pérdidas adversarias puede desestabilizar el extractor si los pesos relativos no están bien equilibrados.
* **Soluciones alcanzadas**: Se estructuró un juego minimax alternado que optimiza por un lado los discriminadores de características y predicciones, y por el otro el extractor y detector mediante un peso ponderador $\alpha$.

### 3. Experiment
Describe el diseño experimental, datasets y configuraciones de red.
* **Problemas atacados**: Validar empíricamente la efectividad de la alineación de predicciones en escenarios de teledetección del mundo real.
* **Limitaciones de ese entonces**: Evaluar modelos de dominio adaptado en datasets de clasificación simple no refleja las dificultades geométricas de la localización de vehículos.
* **Soluciones alcanzadas**: Se utilizó como fuente el dataset satelital COWC (Cars Overhead with Context) y como objetivo imágenes aéreas reales de Tokio, Japón, remuestreadas a 0.3m/píxel y con aumento de datos por rotación.

#### 3.1. Dataset
Detalla las propiedades de las imágenes satelitales utilizadas.
* **Problemas atacados**: El sesgo geográfico y las discrepancias de sensores en teledetección.
* **Limitaciones de ese entonces**: Dataset COWC tiene iluminación y entornos urbanos occidentales, muy diferentes a la densidad y texturas de carreteras en Japón.
* **Soluciones alcanzadas**: El dataset de entrenamiento se estructuró con 6,264 imágenes de la fuente (COWC) y 93,344 imágenes del dominio objetivo sin etiquetas (Tokio) para forzar la transferencia ciega.

#### 3.2. Experimental Setting
Hiperparámetros, backbones y detalles de hardware.
* **Problemas atacados**: Configurar y evaluar de forma reproducible las variantes del pipeline propuesto.
* **Limitaciones de ese entonces**: Las GPUs estándar tienen problemas para procesar minibatches grandes de imágenes de gran resolución.
* **Soluciones alcanzadas**: Se preentrenó SSD-VGG16 en la fuente (40k iteraciones, lote de 32) y se ajustó con DA (15k iteraciones, lote de 64). Se evaluaron tres variantes: sin normalización (w/o norm), normalización en discriminadores y predicciones (norm D and P) y normalización aplicada únicamente al extractor y detector (norm P).

### 4. Result and Discussion
Analiza cuantitativamente las métricas obtenidas.
* **Problemas atacados**: Demostrar la ganancia en precisión absoluta y analizar la interacción de la técnica CWN con los discriminadores.
* **Limitaciones de ese entonces**: La alineación sin pesos CWN apenas mejora la precisión de los vehículos y aplicar CWN al discriminador causa colapso del modelo.
* **Soluciones alcanzadas**: Las pruebas revelaron que la variante `norm P` (CWN aplicada solo a la actualización del extractor y detector) alcanza el mejor rendimiento, mejorando el AP en +5.0% frente al baseline adversario simple y +12.8% frente al modelo sin adaptación de dominio.

### 5. Conclusion
Conclusiones de la investigación y futuras líneas de desarrollo.
* **Problemas atacados**: La validación de la alineación adversaria del clasificador y localizador de detección.
* **Limitaciones de ese entonces**: Determinar si la normalización de pesos CWN propuesta se puede refinar más allá del enfoque heurístico implementado.
* **Soluciones alcanzadas**: Se concluye que alinear el espacio de predicción es altamente efectivo para resolver el cambio de dominio en teledetección. Se propone como trabajo futuro formular un CWN adaptativo más sofisticado y extender el método a detectores de objetos generales en luz visible y multiespectral.
