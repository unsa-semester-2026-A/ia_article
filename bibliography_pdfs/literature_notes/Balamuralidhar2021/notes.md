# MultEYE: Monitoring System for Real-Time Vehicle Detection, Tracking and Speed Estimation from UAV Imagery on Edge-Computing Platforms

- **Key**: Balamuralidhar2021
- **Year**: 2021
- **Venue**: Remote Sensing (MDPI)

## Resumen

Este artículo presenta MultEYE, un sistema integrado de monitoreo de tráfico en tiempo real diseñado para ejecutarse a bordo de vehículos aéreos no tripulados (UAV) en plataformas de computación de borde (edge computing). El framework aborda simultáneamente tres tareas clave: (1) detección de vehículos mediante una red neuronal convolucional (CNN) optimizada bajo aprendizaje multitarea, (2) seguimiento de múltiples objetos (Multi-Object Tracking, MOT) en CPU mediante filtros de correlación adaptativos (MOSSE), y (3) estimación de la velocidad del vehículo en el espacio físico real a partir de parámetros geométricos y de vuelo del UAV sin requerir calibración extrínseca de cámara. La innovación principal en la detección radica en la arquitectura MultEYE, que entrelaza YOLOv4 y ENet en un esquema de aprendizaje multitarea (detección y segmentación semántica) durante el entrenamiento. Al finalizar el entrenamiento, la cabeza de segmentación se desacopla para preservar la velocidad. En la placa NVIDIA Xavier NX, el detector optimizado alcanza 29 FPS a baja resolución, y el pipeline optimizado (detección en el primer cuadro y rastreo en los 9 siguientes) alcanza 33 FPS en resolución de 3K (3072 × 1728 píxeles).

## Secciones y Subsecciones

### 1. Introduction
Introduce el papel creciente de los UAVs para el monitoreo de tráfico frente a las tecnologías tradicionales fijas en tierra (bucles inductivos, CCTV). Destaca la necesidad de procesar los datos a bordo del UAV (edge computing) para evitar la latencia y fallos de transmisión asociados al envío de flujos de video a estaciones terrestres.
* **Problemas atacados**: Falta de sistemas de monitoreo de tráfico en tiempo real basados en UAVs que ejecuten detección, seguimiento y estimación de velocidad integrados de manera local a bordo del dron.
* **Limitaciones de ese entonces**: Alto costo y rango de cobertura limitado de cámaras terrestres fijas; dependencia de la transmisión de datos a gran escala a estaciones terrenas, propensa a caídas y latencia.
* **Soluciones alcanzadas**: Propuesta del sistema MultEYE que procesa todo el pipeline de tráfico en tiempo real a bordo mediante hardware embebido de bajo consumo.

### 2. Related Work
Realiza una revisión metodológica de las bases tecnológicas que componen el sistema MultEYE.
* **Problemas atacados**: Revisar y seleccionar el estado del arte de las tecnologías de segmentación, detección, aprendizaje multitarea, rastreo y velocidad.
* **Limitaciones de ese entonces**: Las soluciones existentes son metodologías aisladas diseñadas para servidores con alta capacidad gráfica, no aptas para UAVs con baterías y cómputo limitados.
* **Soluciones alcanzadas**: Compilación y fundamentación de las técnicas optimizadas que se integran en el diseño modular de MultEYE.

#### 2.1. State-of-the-Art Semantic Segmentation
* **Problemas atacados**: Extraer características de contornos y bordes finos a gran velocidad.
* **Limitaciones de ese entonces**: Las redes FCN tradicionales son computacionalmente costosas y no procesan de manera eficiente la información de contexto global en tiempo real.
* **Soluciones alcanzadas**: Uso de decodificadores y convolutions atrous ligeras (como ENet) para recuperar rasgos espaciales reduciendo parámetros.

#### 2.2. State-of-the-Art Object Detection
* **Problemas atacados**: Detección precisa de vehículos pequeños en imágenes aéreas a alta velocidad.
* **Limitaciones de ese entonces**: Los detectores de dos etapas (Faster R-CNN) son lentos; los de una etapa (YOLO, SSD) fallan al localizar objetos diminutos.
* **Soluciones alcanzadas**: Selección de detectores de una sola etapa rápidos (como YOLOv4) que incorporan agregaciones multiescala (PANet, SPP).

#### 2.3. Multi-Task Learning
* **Problemas atacados**: Incrementar la precisión de detección en objetos pequeños sin penalizar la velocidad de inferencia del hardware de borde.
* **Limitaciones de ese entonces**: Los detectores rápidos tienen bajo desempeño en objetos pequeños, y los esquemas multitarea comunes arrastran el sobrecoste computacional de todas las ramas en inferencia.
* **Soluciones alcanzadas**: Incorporación de una cabeza de segmentación que añade un sesgo inductivo y regularización al backbone durante el entrenamiento, retirándose completamente en fase de inferencia.

#### 2.4. Multi-Object Tracking
* **Problemas atacados**: Seguimiento estable de múltiples vehículos bajo rotación y oclusión parcial desde una cámara aérea.
* **Limitaciones de ese entonces**: Los rastreadores deep (DeepSORT, GOTURN) requieren recursos GPU y compiten directamente con el detector, ralentizando la ejecución global.
* **Soluciones alcanzadas**: Adopción de rastreadores basados en filtros de correlación en CPU (MOSSE) que operan a cientos de FPS sin tocar la GPU.

#### 2.5. Vehicle Speed Estimation
* **Problemas atacados**: Medición de la velocidad del vehículo en coordenadas de la carretera desde un sensor en movimiento.
* **Limitaciones de ese entonces**: Métodos anteriores requieren cámaras estáticas terrestres o perspectivas cenitales estrictas (nadir), fallando ante vuelos oblicuos del UAV.
* **Soluciones alcanzadas**: Modelado dinámico de la escala de la imagen (GSD) dependiente de la posición de los píxeles y del ángulo tilt de la cámara.

### 3. Methodology
Detalla el diseño matemático y lógico de la arquitectura del sistema MultEYE.
* **Problemas atacados**: Integración y sincronización de los tres componentes (detección, rastreo, velocidad) en un único flujo de datos de baja latencia.
* **Limitaciones de ese entonces**: La ejecución sucesiva de tareas pesadas satura los recursos del procesador integrado.
* **Soluciones alcanzadas**: Formulación de un pipeline secuencial modular optimizado hardware-aware (GPU para detección y CPU para rastreo y velocidad).

#### 3.1. System Design
* **Problemas atacados**: Definir la estructura física y lógica de los componentes de MultEYE para hardware integrado.
* **Limitaciones de ese entonces**: Los módulos de red estándar tienen redundancia y alta latencia.
* **Soluciones alcanzadas**: Estructuración física de los flujos de datos donde la GPU corre detección y el procesador CPU corre rastreo/velocidad concurrentemente.

##### 3.1.1. Vehicle Detection Architecture
* **Problemas atacados**: Reducción de parámetros del detector multitarea.
* **Limitaciones de ese entonces**: El backbone Darknet53 estándar de YOLOv4 es muy pesado y lento en hardware embebido.
* **Soluciones alcanzadas**: Rediseño a una versión CSPDarknet53(Lite) con un cuarto del tamaño original, incorporando una capa Space-to-Depth para acelerar la inferencia en GPU y skip-connections al decodificador de segmentación.

##### 3.1.2. Vehicle Tracking—Minimum Output Sum of Squared Error (MOSSE)
* **Problemas atacados**: Mantener la estabilidad del rastreo a bordo.
* **Limitaciones de ese entonces**: El filtro de correlación MOSSE básico es sensible a giros del UAV y variaciones extremas de luz.
* **Soluciones alcanzadas**: Robustecimiento del MOSSE mediante transformaciones de contraste logarítmicas y rotaciones/escalados en línea.

##### 3.1.3. Speed Estimation
* **Problemas atacados**: Medir velocidad en metros por segundo en planos oblicuos dinámicos.
* **Limitaciones de ese entonces**: La perspectiva oblicua distorsiona el tamaño del píxel (GSD variable) a lo largo del cuadro de la imagen.
* **Soluciones alcanzadas**: Algoritmo que calcula el GSD local (GSD Rate) de cada píxel combinando datos de altura de vuelo, focal de cámara, y ángulo de inclinación del UAV.

#### 3.2. Data
* **Problemas atacados**: Falta de conjuntos de datos unificados con anotaciones de detección, segmentación y velocidad para UAV.
* **Limitaciones de ese entonces**: Los datasets existentes están fragmentados y carecen de información de sensores de vuelo del dron.
* **Soluciones alcanzadas**: Anotación de instancias en Aeroscapes y recolección de sets propios (SODA, ciclistas con telemetría de vuelo).

##### 3.2.1. Vehicle Detection and Segmentation Dataset
* **Problemas atacados**: Crear un dataset balanceado para el entrenamiento multitarea de MultEYE.
* **Limitaciones de ese entonces**: Aeroscapes tiene imágenes de muy baja altitud y carece de delimitaciones de instancias de vehículos.
* **Soluciones alcanzadas**: Etiquetado manual de cajas de vehículos sobre Aeroscapes y recopilación de 52 imágenes de alta resolución SODA a 30m, 60m y 120m.

##### 3.2.2. Vehicle Tracking and Speed Estimation Dataset
* **Problemas atacados**: Proveer secuencias de video con valores de velocidad de tierra verdaderos (ground truth).
* **Limitaciones de ese entonces**: Inviabilidad jurídica y operacional de bloquear autopistas para estimar la velocidad real de coches usando UAVs.
* **Soluciones alcanzadas**: Uso de 100 secuencias de KIT AIS y captura de ciclistas con velocímetro GPS como proxy de vehículos para validar estimaciones.

#### 3.3. Experiments
* **Problemas atacados**: Establecer un marco de entrenamiento y validación riguroso en hardware embebido.
* **Limitaciones de ese entonces**: Los benchmarks teóricos en PC de escritorio no capturan el comportamiento térmico y de consumo del hardware del dron.
* **Soluciones alcanzadas**: Protocolo de entrenamiento ADAM con pérdidas gIoU, Focal y Tversky, y pruebas de rendimiento específicas en NVIDIA Xavier NX.

##### 3.3.1. Vehicle Detection
* **Problemas atacados**: Entrenar la red de detección minimizando pérdidas de segmentación y detección acopladas.
* **Limitaciones de ese entonces**: Riesgo de inestabilidad y gradientes explosivos al unificar pérdidas de diferente naturaleza.
* **Soluciones alcanzadas**: Entrenamiento en dos fases (congelando y descongelando backbone) con inicialización ImageNet.

##### 3.3.2. Vehicle Tracking and Speed Estimation
* **Problemas atacados**: Validar la precisión del rastreador MOSSE y el error de velocidad.
* **Limitaciones de ese entonces**: Los errores de localización del detector pueden degradar falsamente el rendimiento del rastreador independiente.
* **Soluciones alcanzadas**: Validación del rastreador usando bounding boxes reales, y uso de un buffer de 8 frames para filtrar ruidos viales de velocidad.

##### 3.3.3. Inference on Jetson Xavier NX
* **Problemas atacados**: Evaluar la latencia física de todo el sistema integrado en el hardware de destino.
* **Limitaciones de ese entonces**: Los modos de potencia del procesador embebido limitan los hilos paralelos disponibles en la GPU.
* **Soluciones alcanzadas**: Pruebas de inferencia del pipeline completo bajo límites de potencia de 10W y 15W y diferentes resoluciones de entrada.

### 4. Results & Discussion

#### 4.1. Vehicle Detection
* **Problemas atacados**: Validar la precisión y capacidad de generalización del detector multitarea.
* **Limitaciones de ese entonces**: YOLOv4 tiende a sobreajustar en Aeroscapes y colapsa a resoluciones muy bajas (IoU 0.132 a 120m).
* **Soluciones alcanzadas**: MultEYE logra mAP de 0.834 (superando a YOLOv4 en 4.8%) y demuestra gran generalización a 120m de altura en SODA (IoU 0.844) gracias al entrenamiento multitarea que enfoca características.

#### 4.2. Vehicle Tracking
* **Problemas atacados**: Velocidad y estabilidad del rastreador de vehículos.
* **Limitaciones de ese entonces**: Algoritmos clásicos (MIL, Boosting) fallan en CPU a tiempo real; rastreadores deep saturan la GPU del dron.
* **Soluciones alcanzadas**: MOSSE en CPU alcanza 227.5 FPS, logrando niveles de precisión excelentes (MOTA 90.91, MOTP 93.29) similares a DeepSORT pero consumiendo recursos de CPU y liberando la GPU.

#### 4.3. Speed Estimation
* **Problemas atacados**: Precisión del velocímetro a bordo w.r.t. la velocidad física real.
* **Limitaciones de ese entonces**: El movimiento dinámico del UAV y vibraciones viales inducen errores y variaciones de velocidad.
* **Soluciones alcanzadas**: El error medio compensado es de 1.13 km/h, superando a los radares de tráfico comerciales estándar que tienen un margen de 3.2 km/h.

#### 4.4. Inference on Jetson Xavier NX
* **Problemas atacados**: Optimizar el rendimiento temporal en hardware de borde para vuelo real.
* **Limitaciones de ese entonces**: Ejecutar la red de detección en cada fotograma del flujo de video limita la velocidad a solo 3.74 FPS en alta resolución.
* **Soluciones alcanzadas**: Optimización de la tasa de actualización de detección para aprovechar la asimetría temporal del pipeline.

##### 4.4.1. Vehicle Detection Inference
* **Problemas atacados**: Rendimiento de la red MultEYE sola en Xavier NX.
* **Limitaciones de ese entonces**: Procesar altas resoluciones en modo de ahorro energético (10W) reduce severamente los FPS.
* **Soluciones alcanzadas**: Inferencia en tiempo real de 29.41 FPS en modo 15W a resolución 512x320.

##### 4.4.2. Complete Pipeline Inference
* **Problemas atacados**: Distribución temporal de la carga de trabajo por algoritmo.
* **Limitaciones de ese entonces**: La detección consume el 98% del tiempo de procesamiento total, siendo el cuello de botella del pipeline.
* **Soluciones alcanzadas**: Medición que demuestra que el rastreo (1.6%) y la velocidad (~0%) en CPU consumen un tiempo despreciable w.r.t. la GPU.

##### 4.4.3. Streaming Optimization
* **Problemas atacados**: Elevar la velocidad del sistema completo a altas resoluciones (resolución 3K).
* **Limitaciones de ese entonces**: Ejecutar detección profunda en cada cuadro satura el hardware.
* **Soluciones alcanzadas**: Estrategia de buffer de 10 imágenes: el detector solo corre en el primer cuadro (inicialización de anclas) y el tracker MOSSE en CPU corre en los 9 cuadros restantes. Logra velocidades de 33.04 FPS en resolución 3072x1728.

### 5. Conclusions
Resume las conclusiones del proyecto de monitoreo.
* **Problemas atacados**: Consolidar la viabilidad de sistemas autónomos de tráfico basados en drones.
* **Limitaciones de ese entonces**: Limitaciones de regulaciones legislativas de vuelo urbano y restricciones de autonomía de baterías de UAVs.
* **Soluciones alcanzadas**: Conclusión que valida a MultEYE como un primer paso relevante y robusto en la integración de redes multitarea desmontables y rastreadores GPU-CPU eficientes para drones.
