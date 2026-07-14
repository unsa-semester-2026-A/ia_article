# Fase 0: Pseudo-Labeling Temporal (03_pseudo_labeling.md)

Este documento detalla las especificaciones del pipeline temporal para el auto-etiquetado y detección de los vehículos estacionados que fueron omitidos en las anotaciones originales de `train.csv`.

---

## 1. Objetivo
Detectar con precisión la ubicación de vehículos estáticos a partir de un modelo de alta sensibilidad (YOLO26s-OBB) entrenado únicamente con las anotaciones en movimiento, aplicando un filtro temporal que compense el movimiento relativo del dron (ego-motion) para evitar falsas detecciones.

---

## 2. Flujo de Trabajo del Auto-Etiquetado

```
       ┌────────────────────────┐
       │   Frames del Clip      │
       └──────────┬─────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ YOLO26s-OBB (Inferencia)     │ <── (Umbral de Confianza = 0.10)
   │  Detecta TODOS los vehículos │
   └──────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────┐
   │   Compensación de Ego-Motion │ <── (Homografía ORB + RANSAC)
   │     del Dron (Inter-Frame)   │
   └──────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ Matching Temporal & Tracking │ <── (Asociación de Centroides)
   └──────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────┐
   │    Cruce con Ground Truth    │ <── (Detecciones sin GT matches)
   └──────────┬───────────────────┘
                  ▼
   ┌──────────────────────────────┐
   │ Clasificación de Estáticos   │ <── (Filtro: delta centroide < 8px
   │                              │      durante >= 10 frames)
   └──────────────────────────────┘
```

---

## 3. Especificaciones de los Módulos

### 3.1 Modelo de Inferencia Exhaustiva
* **Arquitectura:** `YOLO26s-obb` (9.8 millones de parámetros). Se selecciona esta versión para tener un balance ideal entre velocidad de inferencia en la GTX 1070 y capacidad de generalización espacial (recall superior a la versión `nano`).
* **Entrenamiento del Detector Base:** El modelo se entrena durante $50$ épocas a una resolución de $640$ píxeles utilizando el train split de la Fase 01 (datos originales).
* **Configuración de Inferencia:**
  - `conf = 0.10`: Umbral muy bajo para maximizar el recall. Se asume que los falsos positivos se descartarán mediante consistencia temporal.
  - `iou = 0.50`
  - Entrada: Todo el conjunto de imágenes de entrenamiento (54,262 fotogramas).

---

### 3.2 Módulo de Compensación de Ego-Motion
Dado que la cámara está montada en un dron con movimiento sutil (no en un poste fijo), los objetos inmóviles se desplazan en el espacio de píxeles de la cámara. Este módulo estima la transformación geométrica del fondo entre fotogramas consecutivos.

* **Entrada:** Imagen del frame $t$, imagen del frame $t+1$, máscaras de los vehículos detectados en el frame $t$.
* **Proceso de Alineación:**
  1. Convertir ambas imágenes a escala de grises.
  2. Extraer puntos clave y descriptores utilizando el algoritmo **ORB** (orientación y escala rápida).
  3. **Máscara de Fondo:** Excluir del cálculo de puntos clave las regiones correspondientes a los vehículos detectados por YOLO en el frame $t$, evitando que los vehículos en movimiento sesguen la homografía del fondo.
  4. Realizar emparejamiento de descriptores mediante fuerza bruta (`BFMatcher` con distancia Hamming) y retener los mejores emparejamientos mediante el test de RANSAC.
  5. Computar la matriz de homografía $H$ de dimensiones $3\times3$ que minimice el error de reproyección.
* **Criterio de Calidad:**
  - Si el número de emparejamientos inliers es $< 10$, se considera que la homografía falló (ej. oclusión extrema, frame ruidoso). Se asume $H = I$ (matriz identidad) y se registra la alerta.
  - El error de reproyección promedio del fondo debe ser $< 3$ píxeles.

---

### 3.3 Módulo de Asociación Temporal (Tracking Simple)
Este módulo vincula las detecciones de vehículos a lo largo del tiempo dentro de un mismo clip (~50 frames).

* **Entrada:** Detecciones OBB crudas de todos los frames del clip, matrices de homografía inter-frame.
* **Proceso de Asociación:**
  1. Para cada frame $t$ del clip:
     - Tomar un centroide detectado $(cx_{t}, cy_{t})$.
     - Proyectar el centroide al plano del frame $t+1$ aplicando la matriz de homografía:
       
       $$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H_{t \to t+1} \begin{bmatrix} cx_{t} \\ cy_{t} \\ 1 \end{bmatrix}$$
       
       $$cx'_{t} = x'/w', \quad cy'_{t} = y'/w'$$
       
  2. Asociar el centroide proyectado con las detecciones reales en el frame $t+1$ usando una estrategia greedy de distancia mínima.
  3. Si la distancia entre el centroide proyectado y el centroide real en $t+1$ es $< 30$ píxeles, se enlazan como pertenecientes a la misma "instancia temporal".
  4. Mantener un registro del historial de posiciones para cada vehículo enlazado.

---

### 3.4 Clasificador de Estado de Movimiento
Filtra y etiqueta las trayectorias resultantes para aislar los vehículos que no tienen movimiento real.

* **Filtro de Ground Truth:** Si una trayectoria tiene un solapamiento rIoU $\ge 0.3$ con alguna de las cajas de Ground Truth en más del 20% de sus frames, se clasifica automáticamente como **Vehículo en Movimiento** (ya anotado legítimamente en el dataset).
* **Criterio de Estacionado:** Las trayectorias que NO tienen match con el Ground Truth se evalúan bajo las siguientes reglas:
  1. **Duración Mínima:** La instancia debe detectarse y rastrearse durante al menos $10$ frames consecutivos dentro del clip.
  2. **Desplazamiento Residual:** Se calcula la dispersión máxima de los centroides compensados por homografía a lo largo de su trayectoria:
     
     $$\Delta_{\text{motion}} = \max_{i,j \in \text{frames}} \text{distancia}( (cx'_i, cy'_i), (cx'_j, cy'_j) )$$
     
  3. Si $\Delta_{\text{motion}} < 8$ píxeles, el vehículo se clasifica formalmente como **Vehículo Estacionado No Anotado**.

---

## 4. Estructura del Output

El resultado del pseudo-labeling se guarda como un archivo JSON con la siguiente estructura para consumo de la Fase 1 (LaMa):

```json
{
  "v_ab12cd34ef_0000": [
    {"cx": 450.2, "cy": 300.5, "w": 45.0, "h": 22.0, "angle": 35.5, "conf": 0.88},
    {"cx": 120.4, "cy": 890.1, "w": 38.0, "h": 18.0, "angle": 90.0, "conf": 0.76}
  ],
  "v_ab12cd34ef_0001": [
    ...
  ]
}
```

---

## 5. Protocolo de Verificación Manual (Auditoría de Calidad)

Antes de proceder a la remoción masiva con LaMa, se debe validar que el pseudo-labeler tenga una tasa de falsos positivos extremadamente baja para no borrar vehículos en movimiento reales.

1. **Muestreo:** Seleccionar aleatoriamente $50$ clips del conjunto de entrenamiento.
2. **Ejecución:** Correr el pipeline de la Fase 0 sobre las imágenes de estos clips.
3. **Auditoría:** Utilizar una herramienta de visualización simple en Python (ej. script de OpenCV que dibuje las cajas de pseudo-labeling en rojo y las del GT en verde). Un evaluador humano califica cada caja roja como:
   - **Verdadero Estacionado (TP):** Un vehículo parado al borde de la vía sin caja verde.
   - **Falso Estacionado (FP):** Un vehículo en marcha (o detenido momentáneamente por flujo de tráfico/semáforo) que fue marcado erróneamente para borrarse.
4. **Métrica de Control:** La precisión del pseudo-labeling se calcula como:
   
   $$\text{Precision}_{pseudo} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   
5. **Umbral de Aceptación:** $\text{Precision}_{pseudo} \ge 90\%$. Si es inferior, se debe ajustar el parámetro `motion_threshold` (reducir el valor permitido de píxeles) o aumentar el número de frames mínimos estáticos (`min_static_frames`).
