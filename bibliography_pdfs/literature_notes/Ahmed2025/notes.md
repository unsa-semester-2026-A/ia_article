# Automated geolocalization of vehicles from UAV footage: evaluating measurement precision of object detection and segmentation methods

- **Key**: Ahmed2025
- **Year**: 2025
- **Venue**: Applied Geomatics

## Resumen
Este artículo introduce un método de geolocalización automatizada para vehículos a partir de grabaciones de vehículos aéreos no tripulados (UAVs). Combina el algoritmo de emparejamiento de características SIFT para el georreferenciamiento automático con modelos de detección y segmentación de objetos basados en aprendizaje profundo. El estudio se enfoca en determinar la precisión dimensional de los vehículos localizados. Se comparan tres configuraciones del detector de objetos YOLO: YOLOv8 estándar (AABB), un modelo híbrido YOLOv8 + SAM (Segment Anything Model) y YOLOv8-OBB (Oriented Bounding Boxes). La evaluación analiza el impacto de la altitud del dron, la rotación de los vehículos, la variabilidad interna del error (RMSE) y el tiempo de inferencia. Los resultados demuestran que YOLOv8 + SAM y YOLOv8-OBB superan con creces al YOLOv8 estándar al preservar las dimensiones reales de los vehículos y eliminar el error inducido por la rotación. No obstante, por su eficiencia y velocidad de inferencia, la variante YOLOv8-OBB es la opción más práctica para aplicaciones en tiempo real de monitoreo de tráfico y análisis de conflictos de seguridad (como TTC y PET).

## Secciones y Subsecciones

### Introduction
Esta sección introduce la relevancia de los sistemas de monitoreo de tráfico en carreteras (RTM) y el análisis de la seguridad vial dentro de las ciudades inteligentes, justificando el uso de drones (UAVs) e Inteligencia Artificial (IA) para optimizar el flujo de tráfico y automatizar flujos de trabajo.
* **Problemas atacados**: La necesidad de estimar de forma automatizada y precisa la posición espacial y las dimensiones físicas de los usuarios de la vía. Esta precisión es esencial para calcular métricas de seguridad vial basadas en indicadores sustitutos de conflictos, tales como el Tiempo de Colisión (TTC) y el Tiempo Post-Invasión (PET), cuyos umbrales de seguridad dependen críticamente de las posiciones reales y huellas (footprints) de los vehículos.
* **Limitaciones de ese entonces**: Los métodos convencionales de monitoreo de tráfico son costosos e ineficientes. Las aplicaciones previas de visión computacional sufren por la sobreestimación del tamaño de las cajas delimitadoras debido a cambios en la orientación de los vehículos (p. ej., durante giros o cambios de carril) al usar cajas delimitadoras estándar alineadas a los ejes (AABB), lo que introduce fluctuaciones artificiales en las mediciones y genera evaluaciones de riesgo erróneas.
* **Soluciones alcanzadas**: Se propone evaluar experimentalmente múltiples modelos de detección y segmentación en entornos del mundo real. El estudio introduce un flujo que integra el algoritmo SIFT para un georreferenciamiento automatizado preciso y evalúa tres configuraciones basadas en YOLOv8 (estándar, híbrida con SAM, y OBB) para contrarrestar la sobreestimación de dimensiones de las cajas delimitadoras.

### Related works
Revisión de la literatura científica existente sobre georreferenciación de imágenes de UAV, detección de objetos con redes neuronales convolucionales y geolocalización en espacio geográfico.
* **Problemas atacados**: La calibración de píxeles de video a coordenadas del mundo real (georeferenciación) de manera dinámica, y la detección/seguimiento eficiente de vehículos con alta precisión espacial.
* **Limitaciones de ese entonces**: El georreferenciamiento de datos de UAV es un desafío debido al movimiento dinámico de la plataforma. Los métodos manuales son lentos y propensos a errores. Las arquitecturas de detección de objetos basadas en YOLO históricamente emplean cajas AABB que carecen de ajuste angular, sobredimensionando el área del objeto. Asimismo, la mayor parte de las investigaciones previas se concentran puramente en la exactitud del conteo y clasificación (métricas como mAP) mas no en evaluar la precisión de las dimensiones físicas de los vehículos geolocalizados.
* **Soluciones alcanzadas**: Se analiza la integración de localización visual absoluta (AVL) mediante SIFT con mínimos cuadrados (LSM) para georreferenciación en segundos. Se examina la efectividad de YOLOv4 con DeepSORT para rastreo de vehículos y modelos YOLOv8 aplicados a ortofotos GeoTIFF para geolocalización de daños en pavimentos. Se destaca el uso de YOLOv8-OBB (cajas delimitadoras orientadas) entrenado en el dataset DOTA como un método viable para estimar el ángulo de rotación de los objetos y obtener dimensiones reales.

### Materials and methods
Presentación del marco metodológico propuesto para la adquisición de datos, calibración espacial de las imágenes y la implementación de los tres modelos de detección evaluados.
* **Problemas atacados**: Diseñar un flujo de trabajo que permita la geolocalización métrica de vehículos y el análisis dimensional con alta precisión a partir de grabaciones de UAV.
* **Limitaciones de ese entonces**: La inestabilidad y la distorsión de perspectiva en videos tomados desde drones en movimiento que impiden mediciones métricas estables.
* **Soluciones alcanzadas**: Se propone un marco de trabajo que comienza con la creación de un ortomosaico de referencia en proyección UTM usando WebODM. Se obtienen grabaciones en Hasselt, Bélgica, a altitudes de 60 y 90 metros con un dron DJI Mini 3 Pro. Usando feature-matching y SIFT se calcula una matriz de homografía para transformar píxeles en coordenadas UTM. Luego, se procesan los fotogramas con tres configuraciones de YOLOv8 (YOLOv8 estándar, YOLOv8 + SAM, y YOLOv8-OBB) para derivar dimensiones físicas que se contrastan con las mediciones reales de referencia en QGIS.

#### Calibration of UAV frames in projected coordinate system using SIFT
Esta subsección describe detalladamente el proceso matemático y computacional para automatizar la transformación de coordenadas de píxeles a coordenadas del mundo real UTM.
* **Problemas atacados**: Convertir coordenadas de píxeles bidimensionales en coordenadas UTM tridimensionales del mundo real para estimar parámetros del flujo de tráfico espacial sobre el tiempo de forma automatizada.
* **Limitaciones de ese entonces**: Los métodos basados en GPS o en localización visual relativa (RVL) sufren acumulación de errores y derivas (drift) temporales.
* **Soluciones alcanzadas**: Se implementa el algoritmo SIFT (detección de extremos en escala-espacio usando Diferencia de Gaussianas (DoG), localización de puntos clave, asignación de orientación para invarianza rotacional y descriptores de 128 elementos). Se emplea la prueba de ratio de Lowe y RANSAC para eliminar puntos de correspondencia ruidosos. Usando el ortomosaico de referencia generado previamente a 90m de altura en orientación top-down (nadir), se calcula una matriz de homografía estable que proyecta las coordenadas del video directamente a la rejilla UTM del mundo real, eliminando el drift.

#### Vehicle detection and localization
Esta subsección detalla los métodos utilizados para detectar los vehículos, realizar su seguimiento a lo largo de los fotogramas y loguear sus posiciones espaciales.
* **Problemas atacados**: Mantener la identificación única y continua de los vehículos a lo largo del tiempo de forma automatizada para reconstruir sus trayectorias dimensionales.
* **Limitaciones de ese entonces**: La pérdida de objetos durante oclusiones temporales, lo que interrumpe el análisis continuo de seguridad vial y divide los tracks de los vehículos.
* **Soluciones alcanzadas**: Se integra YOLOv8 con el algoritmo de rastreo Bot-SORT, el cual permite la re-identificación de vehículos incluso tras oclusiones temporales. Esto facilita el registro de las cuatro esquinas de la caja delimitadora en formato GeoJSON para su mapeo y análisis directo en la plataforma QGIS.

#### 8th generation you only look once (YOLO) architecture
Detalles sobre el funcionamiento del detector de una sola etapa YOLO en su octava generación.
* **Problemas atacados**: Detección rápida y en tiempo real de múltiples clases de vehículos desde una perspectiva aérea.
* **Limitaciones de ese entonces**: Modelos de detección convencionales que no están entrenados para las características de escala y resolución de imágenes capturadas por drones (UAV).
* **Soluciones alcanzadas**: Se utilizan los pesos de YOLOv8 entrenados en el conjunto de datos VisDrone2019-DET, logrando una precisión promedio AP@0.5 del 64% para la clase "carro", lo cual permite una detección estable y continua en todas las tramas de video evaluadas.

#### Segment anything model (SAM)
Descripción de la integración del modelo de segmentación de Meta AI con YOLOv8.
* **Problemas atacados**: Refinar el área de las cajas delimitadoras para ajustarse estrictamente al contorno real del vehículo y posibilitar la rotación de la caja.
* **Limitaciones de ese entonces**: El modelo SAM por sí solo requiere prompts o especificaciones de entrada y es extremadamente demandante en términos computacionales, lo que limita su uso en flujos de video directos.
* **Soluciones alcanzadas**: Se desarrolla un modelo híbrido YOLOv8 + SAM. YOLOv8 detecta el vehículo y genera la caja delimitadora inicial, que sirve de prompt para que SAM extraiga la máscara de segmentación detallada. Luego, se calcula el contorno máximo y el rectángulo de área mínima que encierra dicho contorno, derivando así una caja delimitadora rotada y ajustada de alta precisión dimensional sin necesidad de entrenamiento adicional.

#### YOLOv8 oriented bounding boxes
Descripción de la variante de cajas orientadas (OBB) en arquitecturas YOLO recientes.
* **Problemas atacados**: Predecir de forma directa y nativa el ángulo de orientación del vehículo durante el proceso de detección para evitar la sobreestimación dimensional de las cajas AABB.
* **Limitaciones de ese entonces**: Las cajas delimitadoras alineadas a los ejes (AABB) de YOLOv8 estándar carecen de capacidad angular, abarcando una gran cantidad de área de asfalto vacía cuando el vehículo está girado.
* **Soluciones alcanzadas**: Se emplea un modelo YOLOv8-OBB entrenado en el dataset DOTA, el cual introduce un parámetro angular directamente en la salida del modelo. Esto permite estimar la orientación exacta del vehículo y reducir drásticamente el espacio sobrante en la caja delimitadora.

### Results and discussion
Análisis y comparación cuantitativa del rendimiento de los tres modelos implementados bajo condiciones de altitud y rotación.
* **Problemas atacados**: Validar experimentalmente la exactitud de las dimensiones estimadas por los modelos frente a las dimensiones reales obtenidas sobre el terreno.
* **Limitaciones de ese entonces**: La falta de estudios comparativos cuantitativos sobre la sensibilidad dimensional de las cajas delimitadoras frente a la orientación y altitud de vuelo.
* **Soluciones alcanzadas**: Se procesan 300 fotogramas de video tomados a 60m y 90m de altitud, analizando 5,100 y 21,900 cajas delimitadoras respectivamente. Las áreas estimadas en UTM se comparan con las mediciones reales de QGIS para calcular el Root Mean Square Error (RMSE) de cada trayectoria.

#### Experimental findings
Presentación detallada de los resultados del error y las variaciones internas de los modelos.
* **Problemas atacados**: Cuantificar la desviación dimensional de las áreas calculadas por cada algoritmo.
* **Limitaciones de ese entonces**: Las fluctuaciones internas de los modelos que introducen inconsistencia en la estimación métrica del tamaño de los vehículos.
* **Soluciones alcanzadas**: El análisis reveló que YOLOv8 estándar posee el peor rendimiento con un RMSE medio de 3.31 m² y alta variabilidad (SD = 2.41). Por el contrario, YOLOv8-OBB exhibió la precisión y consistencia más altas, con un RMSE medio de solo 0.99 m² y baja variabilidad (SD = 0.93), seguido de cerca por YOLOv8 + SAM con un RMSE medio de 1.32 m² (SD = 0.87).

#### Impact of UAV altitude and vehicle rotation on localization
Análisis estadístico de la influencia de la altura de vuelo y el ángulo de giro de los vehículos en la exactitud métrica.
* **Problemas atacados**: Determinar de forma rigurosa si la altitud o la orientación física de los vehículos deterioran de manera significativa el RMSE.
* **Limitaciones de ese entonces**: La falta de pruebas de significancia estadística en los errores de geolocalización que consideren varianzas desiguales entre grupos.
* **Soluciones alcanzadas**: Se ejecuta la prueba t de Welch. Respecto a la altitud (60m vs 90m), no se encontraron diferencias significativas de RMSE para ninguno de los modelos (p > 0.05). Respecto a la rotación de los vehículos, el test reveló que YOLOv8 estándar es altamente sensible a la orientación (p-value = 0.0036), mientras que YOLOv8 + SAM (p = 0.32) y YOLOv8-OBB (p = 0.62) son robustos e inmunes a la rotación. Sin embargo, SAM es 382 veces más lento en inferencia CPU que YOLOv8 estándar, haciendo inviable su uso en tiempo real, mientras que YOLOv8-OBB es solo 1.5 veces más lento, resultando ser el método más práctico y apto.

### Conclusion
Resumen de las contribuciones, viabilidad técnica y limitaciones del estudio.
* **Problemas atacados**: Establecer directrices sobre la implementación de tecnologías basadas en UAV e IA en flujos reales de transporte y seguridad vial.
* **Limitaciones de ese entonces**: Las restricciones operativas de los UAVs, tales como la corta duración de la batería del dron y la dependencia absoluta de condiciones de luz diurna y cielo despejado.
* **Soluciones alcanzadas**: Se concluye que YOLOv8-OBB es la configuración óptima para aplicaciones de seguridad vial de monitoreo automático debido a su inmunidad a la rotación, bajo error métrico y velocidad de procesamiento competitiva.
