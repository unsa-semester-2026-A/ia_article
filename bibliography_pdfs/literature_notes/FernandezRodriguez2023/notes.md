# Automated detection of vehicles with anomalous trajectories in traffic surveillance videos

- **Key**: FernandezRodriguez2023
- **Year**: 2023
- **Venue**: Integrated Computer-Aided Engineering

## Resumen
Este artículo presenta un modelo automatizado, no supervisado y dinámico (que opera fotograma a fotograma en tiempo real con datos históricos y actuales) para la detección de vehículos que realizan maniobras o trayectorias anómalas, tales como la conducción en sentido contrario (counterflow) o la marcha atrás en vías rápidas, a partir de cámaras de vigilancia de tráfico. La metodología propuesta consta de tres etapas secuenciales: (1) Detección de vehículos frame a frame empleando una red YOLOv5 adaptada para retornar únicamente clases vehiculares; (2) Seguimiento de vehículos (tracking) mediante un algoritmo de asignación lineal (LSAP) personalizado que calcula el costo basado en la distancia euclidiana de centroides y optimiza la asociación con heurísticas de tamaño adaptativo y una métrica de similitud semántica entre el contenido visual de las cajas delimitadoras (obtenida mediante vectores de características de redes ResNet/VGG con distancia de coseno); y (3) Procesamiento de trayectorias, que estima vectores de velocidad instantánea, identifica los $N=5$ vecinos espaciales más cercanos del vehículo en una ventana temporal y calcula una métrica de anomalía basada en la diferencia de velocidad. Dicha métrica se suaviza temporalmente con un filtro de mediana y se contrasta con percentiles históricos adaptativos. Evaluado en 237 videos viales de bases de datos reales y sintéticas (incluyendo CARLA, Ko-PER, CDnet e Nvidia AI City Challenge), el método supera a rastreadores de la literatura como SORT y BYTE, demostrando robustez frente a perturbaciones climáticas de lluvia y nieve.

## Secciones y Subsecciones

### 1. Introduction
Se introduce el gran crecimiento en la instalación de cámaras de videovigilancia vial y la necesidad de sistemas autónomos de detección de anomalías de tráfico para mitigar accidentes de tráfico y conductas de conducción peligrosas.
* **Problemas atacados**: La necesidad de detectar de manera automática y no supervisada comportamientos viales anómalos de alta peligrosidad, especialmente vehículos conduciendo en sentido contrario (wrong-way) o retrocediendo en autopistas principales, para alertar rápidamente a operadores humanos.
* **Limitaciones de ese entonces**: A pesar del gran desarrollo en la detección y clasificación genérica de vehículos, la detección automática de trayectorias en sentido contrario ha recibido escasa atención. Los enfoques previos suelen requerir calibraciones complejas de la escena a priori, suponer orientaciones de tráfico estáticas específicas, o evaluar las trayectorias de forma retrospectiva (post facto) al terminar el video, impidiendo alertas tempranas.
* **Soluciones alcanzadas**: Se propone un modelo dinámico e independiente de la orientación que evalúa el tráfico fotograma a fotograma. Optimiza el tracking resolviendo fallos en vehículos lejanos y camiones articulados, y define un indicador de anomalía dinámico basado en diferencias de velocidad local respecto a vecinos espaciotemporales contiguos.

### 2. Related work
Revisión de la literatura científica sobre detección de objetivos (monofásicos y bifásicos), estimación de trayectorias y análisis de anomalías viales.
* **Problemas atacados**: La clasificación y seguimiento en sistemas de transporte inteligente (ITS), y la detección de maniobras inusuales en base a patrones viales previos.
* **Limitaciones de ese entonces**: Las técnicas clásicas de trayectoria basadas en flujo óptico y sustracción de fondo requieren cámaras extremadamente estables y largos tiempos de entrenamiento para cada escena individual, fallando ante variaciones climáticas. Los métodos modernos basados en agrupamiento (clustering) de trayectorias (ej. SVM de una clase, agrupamiento jerárquico o autoencoders LSTM) son offline o sufren de inestabilidad frente al ruido de detección, generando una gran cantidad de falsos positivos en el tracking.
* **Soluciones alcanzadas**: Se desarrolla una tubería no supervisada que aprovecha detectores profundos genéricos (YOLOv5) acoplados a una lógica de seguimiento resistente a fluctuaciones del detector y que realiza comprobaciones locales instantáneas en lugar de clasificaciones globales de trazo completo.

### 3. Methodology
Presenta detalladamente el diseño arquitectónico de tres módulos: detección de vehículos, seguimiento (tracking) y procesamiento de trayectorias.
* **Problemas atacados**: Integrar detección, tracking y física cinemática en un modelo acoplado y parametrizable para CCTV.
* **Limitaciones de ese entonces**: La pérdida de anclaje de tracks en vehículos con un tamaño visual pequeño y la estimación errónea de velocidades en la aparición/desaparición de vehículos en los bordes de la cámara.
* **Soluciones alcanzadas**: Se estructura el flujo de datos frame a frame. El video se procesa con YOLOv5 para derivar coordenadas, se asocian de manera secuencial aplicando restricciones de distancia física y similitud visual de parches, y finalmente se derivan velocidades y anomalías locales.

#### 3.1. Vehicle detection
* **Problemas atacados**: Localización espacial eficiente y detección robusta de vehículos en cada fotograma del flujo de video.
* **Limitaciones de ese entonces**: Los detectores devuelven gran cantidad de clases irrelevantes (peatones, semáforos, animales) y a menudo confunden cabinas de camiones con automóviles individuales.
* **Soluciones alcanzadas**: Se implementa YOLOv5 configurado para extraer únicamente clases vehiculares (carro, moto, bus, camión). Para resolver errores de clasificación interna de camiones grandes, se descarta la etiqueta de clase antes del algoritmo de Supresión No Máxima (NMS) para procesar únicamente las geometrías de las cajas $S''$.

#### 3.2. Vehicle tracking
* **Problemas atacados**: Asociar cajas delimitadoras de fotogramas sucesivos para reconstruir el camino del vehículo superando fallos de detección.
* **Limitaciones de ese entonces**: Los rastreadores populares como SORT y BYTE calculan el costo de emparejamiento basándose en el solapamiento IoU. Si un vehículo de tamaño pequeño en la distancia se desplaza rápido, las cajas consecutivas no se intersectan (IoU=0), quebrando la trayectoria. Adicionalmente, el detector puede fallar un solo frame o aislar la cabina del camión, causando brincos falsos de trayectoria que simulan altas velocidades anormales.
* **Soluciones alcanzadas**: Se propone un seguidor personalizado basado en el problema de asignación lineal (LSAP) cuyo costo es la distancia euclidiana de centroides. Incorpora heurísticas: (1) no se asocian cajas cuyas proporciones de tamaño dimensional cambien abruptamente; (2) la distancia máxima permitida entre centroides es adaptativa al tamaño del vehículo (1.25 veces la menor dimensión de la caja para vehículos grandes y 0.75 veces para lejanos o pequeños); (3) se introduce similitud de parches visuales. Se redimensionan los parches de las cajas a una red CNN (ResNet/VGG) para extraer embeddings y calcular su distancia de coseno, prohibiendo asociaciones si la similitud supera un umbral para evitar mezclar vehículos adyacentes homogéneos.

#### 3.3. Trajectory processing
* **Problemas atacados**: Cuantificar la desviación cinemática de un vehículo para identificar si circula en sentido contrario o realiza marcha atrás temeraria.
* **Limitaciones de ese entonces**: El cálculo del centro de la caja como posición física induce un error cuando un vehículo entra de forma gradual a la escena (el bounding box crece de forma asimétrica, simulando una velocidad lenta falsa). Esto afecta especialmente a camiones grandes en los bordes de la imagen. Además, los fallos de tracking introducen picos de velocidad (ruido de alta frecuencia) que actúan como falsos positivos de anomalía.
* **Soluciones alcanzadas**: Se calcula la velocidad instantánea como la diferencia de posición. Para cada vehículo, se localizan los $N=5$ vecinos espaciales más cercanos en una ventana temporal $F=60$ frames. La anomalía $A_i(t)$ es la media de los módulos de la diferencia de velocidad. El ruido se suaviza mediante un filtro de mediana temporal de 3 fotogramas. Para evitar el sesgo de velocidad en bordes de entrada/salida de la cámara, se aplica "corrección de borde": se anula la alerta si alguna esquina de la caja está a una distancia menor del 5% de su lado más pequeño respecto al borde del marco visual. El vehículo se etiqueta como anómalo si la anomalía supera en un factor de escala $s$ al percentil $P_k$ histórico o si mantiene estados sospechosos durante más de 60 fotogramas consecutivos.

### 4. Experiments and results
Valida el rendimiento del sistema sobre conjuntos de datos controlados y abiertos utilizando métricas de detección y estadística no paramétrica.
* **Problemas atacados**: Sintonizar los hiperparámetros del detector de anomalías y comparar el algoritmo propuesto con rastreadores de referencia en la literatura.
* **Limitaciones de ese entonces**: La escasez de bases de datos públicas de CCTV etiquetadas con vehículos que viajen físicamente en sentido contrario.
* **Soluciones alcanzadas**: Se estructuran cuatro conjuntos de datos con 237 videos en total (reales, CARLA, Ko-PER, CDnet y Nvidia AI City Challenge) anotando manualmente los frames anómalos. Se evalúan precisión ($P$), recall ($R$) e índice de Jaccard ($J$).

#### 4.3. Results
* **Problemas atacados**: Analizar el comportamiento dimensional y estadístico del modelo ante variaciones climáticas e hiperparámetros.
* **Limitaciones de ese entonces**: La sintonización manual combinada de múltiples parámetros es ineficiente y puede carecer de validez estadística si no se evalúa a nivel de fotogramas.
* **Soluciones alcanzadas**: Se adopta un esquema de sintonización secuencial.

##### 4.3.1. Finding optimal values for s and Pk
* **Problemas atacados**: Encontrar el balance óptimo del factor de escala $s$ y el percentil $P_k$.
* **Limitaciones de ese entonces**: Seleccionar un umbral excesivamente permisivo enmascara las anomalías verdaderas, mientras que un umbral estricto eleva los falsos positivos.
* **Soluciones alcanzadas**: Evaluando combinaciones sin heurísticas avanzadas, se determinó que la mejor configuración que asegura detectar todos los videos con anomalías (cero falsos negativos a nivel de video) es una escala de severidad $s=4$ y un percentil $P_k = 95\%$ ($P_{95}$), logrando la mayor precisión y Jaccard promedio.

##### 4.3.2. Effects of border correction and bounding box similarity
* **Problemas atacados**: Evaluar el impacto de la corrección de borde y la similitud visual en el tracking.
* **Limitaciones de ese entonces**: Filtrar la similitud de cajas de todos los vehículos indiscriminadamente puede romper la continuidad de las trayectorias de vehículos grandes debido a distorsiones ópticas o cambios de perspectiva.
* **Soluciones alcanzadas**: Los experimentos demostraron que aplicar la similitud de parches visuales (ResNet/VGG) únicamente a vehículos de tamaño visual pequeño (menores al 2.5% del lado de la imagen) con un umbral de coseno $T=0.4$ y el modelo VGG11, combinado con la corrección de borde, eleva drásticamente el índice de Jaccard a 0.316 y la precisión a 0.550, superando al modelo base.

##### 4.3.3. Best configuration
* **Problemas atacados**: Visualizar y validar de forma individual los resultados del mejor modelo.
* **Limitaciones de ese entonces**: Los falsos positivos iniciales ocurren en los primeros frames de la secuencia al no contar con suficiente historial de trayectorias.
* **Soluciones alcanzadas**: El modelo configurado con $s=4$, $P_{95}$, corrección de borde y similitud VGG11 con $T=0.4$ para vehículos pequeños detectó correctamente todas las maniobras anómalas reales (marcha atrás en Video1, contravía en Video2/4 y auto detenido en Ko-PER seq. 2) mapeando las interconexiones a los 5 vecinos más cercanos en QGIS.

##### 4.3.4. Comparison with other models
* **Problemas atacados**: Comparar la propuesta frente a los rastreadores estándar de la industria SORT y BYTE.
* **Limitaciones de ese entonces**: SORT y BYTE sufren de derivas y falsos saltos cinemáticos al no validar la similitud semántica de las cajas ni contar con restricciones de distancia adaptativa.
* **Soluciones alcanzadas**: Los resultados demostraron que el rastreador propuesto (Jaccard = 0.606, Precisión = 0.619) supera ampliamente a SORT (Jaccard = 0.567) y BYTE (Jaccard = 0.512) utilizando corrección de borde. Para validar la significancia, se aplicó la prueba Q de Cochran sobre los 418,158 fotogramas procesados, obteniendo p-valores de 0, lo que confirma una superioridad estadísticamente significativa con máxima confianza.

##### 4.3.5. Rain and snow
* **Problemas atacados**: Validar la robustez del modelo ante interferencias ambientales típicas de cámaras de tráfico en exteriores.
* **Limitaciones de ese entonces**: El ruido visual de la lluvia y nieve degrada la calidad del detector YOLOv5 e induce fallos de tracking.
* **Soluciones alcanzadas**: Se introducen lluvia y nieve sintéticas a todos los videos usando la biblioteca `imgaug`. La evaluación demostró una degradación moderada (Jaccard cae de 0.606 a 0.572 con lluvia y 0.546 con nieve), probando que el estimador de anomalía y el seguidor toleran la pérdida parcial de detecciones.

##### 4.3.6. Limitations
* **Problemas atacados**: Exponer honestamente los límites operativos del sistema propuesto.
* **Limitaciones de ese entonces**: (1) El modelo requiere tasas de fotogramas altas; framerates bajos (1/2 o 1/3) causan fallos de tracking en objetos veloces elevando falsos negativos. (2) Dependencia de la calidad del detector YOLOv5 (falla ante baja iluminación nocturna, desenfoque y vehículos extremadamente pequeños o superpuestos). (3) Velocidad de cómputo lenta (una secuencia de 1 minuto toma 3.5 minutos en CPU i7 con GPU RTX 3080 debido a la implementación secuencial en Python), imposibilitando el tiempo real.

### 5. Conclusions
* **Problemas atacados**: Resumir las aportaciones e identificar mejoras futuras.
* **Limitaciones de ese entonces**: El modelo actual asume trayectorias normales monomodales (carreteras de un sentido o flujo único), colapsando en intersecciones donde el tráfico normal sigue múltiples direcciones válidas (distribuciones de velocidad multimodales).
* **Soluciones alcanzadas**: Se confirma la viabilidad del modelo no supervisado para prevención vial. Como trabajo futuro se plantea: (1) modelar intersecciones mediante mezclas probabilísticas multimodales para agrupar direcciones permitidas; (2) optimizar la ejecución traduciendo las redes a TensorRT en C++ para alcanzar tiempo real; y (3) sintonizar el detector para entornos nocturnos y videos de baja tasa de frames.
