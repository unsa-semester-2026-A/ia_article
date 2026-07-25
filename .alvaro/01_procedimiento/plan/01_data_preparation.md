# Preparación de Datos (01_data_preparation.md)

Este documento detalla las especificaciones de diseño, validaciones estadísticas y flujos de trabajo para la carga, estructuración, partición de clips y exportación de datos al formato de entrenamiento de YOLO OBB.

---

## 1. Objetivo
Parsear y limpiar la metadata de `train.csv`, validar las coherencias geométricas y estadísticas del dataset, realizar una partición a nivel de clip con semilla fija para evitar fugas de datos y construir el pipeline de conversión a etiquetas YOLO OBB de 4 puntos normalizados.

---

## 2. Especificación de los Módulos de Procesamiento

### 2.1 Módulo: Parser y Auditor Estadístico
El objetivo de este módulo es procesar el archivo `train.csv` (19.4 MB) y generar un formato estructurado en memoria (DataFrame).

* **Entrada:** Ruta absoluta del archivo `train.csv`.
* **Proceso de Parseo:**
  1. Dividir las filas usando `,` como delimitador de columnas (`Id`, `Target`).
  2. El campo `Id` tiene el formato `v_[video_hash]_[frame_idx]`. Extraer el identificador del clip `v_[video_hash]` y el índice secuencial entero `frame_idx`.
  3. Si `Target == "none"`, generar un registro especial con coordenadas y clase como nulos (`None`) indicando que el frame está vacío.
  4. Si `Target != "none"`, separar las anotaciones individuales por el carácter `;`.
  5. Cada anotación se compone de 6 campos de texto separados por espacios: `category_id cx cy width height angle_deg`. Convertir los campos numéricos a tipo punto flotante, a excepción de `category_id` que se convierte a entero.
* **Validación de Integridad Geométrica:**
  - `category_id` debe pertenecer al rango cerrado $[1, 9]$.
  - `width` y `height` deben ser estrictamente positivos ($>0$).
  - `angle_deg` debe pertenecer al rango $[0, 360)$.
  - Las coordenadas `cx` y `cy` deben ser finitas y no nulas (a menos que el frame sea `none`).
* **Verificación de Correspondencia:**
  - Comprobar que los IDs del parser correspondan exactamente 1 a 1 con los archivos JPG dentro de la carpeta `train/images/`. Ningún archivo huérfano.

#### Resumen Estadístico de Aceptación (dataset.tex)
Al finalizar el parseo, los contadores globales del DataFrame resultante deben coincidir exactamente con los siguientes valores teóricos:
* **Frames Totales:** $54,262$
* **Objetos OBB Totales:** $601,934$
* **Clips Únicos:** $1,088$
* **Frames vacíos (none):** $3,394$
* **Frames con 1 objeto:** $6,212$
* **Frames con ≥2 objetos:** $44,656$
* **Objetos máximos en un solo frame:** $53$
* **Distribución de instancias por Clase:**
  - Clase 1 (Car): $481,731$ ($80.03\%$)
  - Clase 2 (Combi): $10,152$ ($1.69\%$)
  - Clase 3 (Microbus): $2,802$ ($0.47\%$)
  - Clase 4 (Minibus): $18,941$ ($3.15\%$)
  - Clase 5 (Omnibus): $2,283$ ($0.38\%$)
  - Clase 6 (Articulated bus): $250$ ($0.04\%$)
  - Clase 7 (Truck): $32,668$ ($5.43\%$)
  - Clase 8 (Mototaxi): $5,539$ ($0.92\%$)
  - Clase 9 (Motorcycle): $47,568$ ($7.90\%$)

---

### 2.2 Módulo: Partición Libre de Fuga (Clip Splitter)
El objetivo de este módulo es dividir los clips de forma que se garantice que ningún frame de un clip de entrenamiento termine en el conjunto de validación.

* **Entrada:** DataFrame parseado.
* **Proceso de Split:**
  1. Agrupar el DataFrame por la columna `clip_id` y obtener la lista de los $1,088$ clips únicos de entrenamiento.
  2. Aplicar un algoritmo de división determinista con semilla fija utilizando `random_state=42`.
  3. Dividir los clips en una proporción de 80% para entrenamiento y 20% para validación.
  4. Retornar dos listas de identificadores: `train_clips` (~870 clips) y `val_clips` (~218 clips).
  5. Filtrar el DataFrame original para obtener `df_train` y `df_val` según la presencia de sus `clip_id` en las listas.
* **Control de Calidad de la Distribución:**
  - Calcular la proporción relativa de cada clase en el conjunto de entrenamiento vs validación.
  - La proporción porcentual de cada una de las 9 clases en `df_train` y `df_val` no debe diferir en más de un 2% absoluto de su valor en el conjunto completo (ej. si Car es $80.03\%$ en el total, debe estar en el rango de $78.03\% - 82.03\%$ en ambos splits).

---

### 2.3 Módulo: Conversión a YOLO-OBB (Generador de Labels)
Este módulo exporta los OBB en formato de píxeles paramétricos del dataset a la representación de YOLO OBB requerida para el entrenamiento.

* **Entrada:** `df_train` y `df_val`, rutas de salida.
* **Conversión Paramétrica a 4 Puntos:**
  Dada una anotación $(cx, cy, w, h, \theta)$ en píxeles, donde $\theta$ está en grados medidos en sentido antihorario respecto al eje X horizontal de la imagen:
  
  1. Calcular las coordenadas locales del rectángulo centrado en el origen $(0,0)$:
     - $v_1 = (-w/2, -h/2)$
     - $v_2 = (w/2, -h/2)$
     - $v_3 = (w/2, h/2)$
     - $v_4 = (-w/2, h/2)$
  
  2. Rotar cada vértice por un ángulo $\theta$ en radianes:
     - $x'_{i} = x_i \cos(\theta) - y_i \sin(\theta)$
     - $y'_{i} = x_i \sin(\theta) + y_i \cos(\theta)$
  
  3. Trasladar al centro $(cx, cy)$ en la imagen:
     - $x_i^{global} = x'_{i} + cx$
     - $y_i^{global} = y'_{i} + cy$
  
  4. Normalizar todas las coordenadas por el ancho de la imagen $W$ y alto $H$ (valores resultantes en el rango $[0.0, 1.0]$):
     - $x_i^{norm} = x_i^{global} / W$
     - $y_i^{norm} = y_i^{global} / H$

* **Especificaciones del Formato de Salida YOLO:**
  - Crear un archivo de texto `.txt` por cada imagen. Si la imagen es `v_ab12_0000.jpg`, el archivo debe llamarse `v_ab12_0000.txt`.
  - Si la imagen no contiene objetos (`Target == "none"`), el archivo `.txt` debe ser creado y guardado pero debe quedar completamente **vacío**.
  - Si contiene objetos, cada fila del archivo representa una detección con la siguiente estructura (separado por espacios):
    ```
    class_idx x1 y1 x2 y2 x3 y3 x4 y4
    ```
  - **Conversión de Clase 0-indexada (CRÍTICO):** La base de datos original usa identificadores de clase del 1 al 9. YOLO requiere clases indexadas en 0 (0 al 8). **Se debe restar exactamente 1** a cada `category_id` (ej. Car pasa de $1$ a $0$, Motorcycle de $9$ a $8$).

---

## 3. Estructura de Directorios Resultante

```
dataset/
├── train/
│   ├── images/
│   │   ├── v_ab12cd34ef_0000.jpg
│   │   └── ... (copias o symlinks de ~43,400 imágenes)
│   └── labels/
│       ├── v_ab12cd34ef_0000.txt (labels en formato OBB normalizado de 4 puntos)
│       └── ...
└── val/
    ├── images/
    │   ├── v_zy98xw76vu_0000.jpg
    │   └── ... (copias o symlinks de ~10,862 imágenes)
    └── labels/
        ├── v_zy98xw76vu_0000.txt
        └── ...
```

---

## 4. Archivo de Configuración: `smart_dataset.yaml`

El archivo de configuración debe guardarse en la raíz del proyecto y debe tener exactamente la siguiente estructura de rutas relativas:

```yaml
# smart_dataset.yaml
path: /home/alvaro9rqc/1_Pacha/1-unsa/7_S/ia/article/dataset  # Ruta absoluta al dataset
train: train/images
val: val/images

# Mapeo de Clases 0-indexadas
names:
  0: auto
  1: combi
  2: microbus
  3: minibus
  4: omnibus
  5: articulado
  6: camion
  7: mototaxi
  8: motocicleta
```

---

## 5. Criterios de Aceptación y Pruebas Unitarias
- [ ] El parser no debe lanzar ninguna excepción al leer las 54,262 líneas de `train.csv`.
- [ ] La suma del conteo de instancias de todas las clases del parseo debe dar exactamente $601,934$.
- [ ] La lista de clips únicos resultantes de la división de datos debe ser exactamente de $1,088$, dividida en $870$ train clips y $218$ val clips.
- [ ] Al visualizar un frame arbitrario dibujando las cajas OBB a partir de los puntos reconstruidos del archivo `.txt`, estas deben solaparse visualmente con el vehículo con precisión milimétrica sin desvíos de rotación.
- [ ] El script de validación verifica que todos los valores flotantes de coordenadas en los archivos `.txt` pertenezcan estrictamente al rango $[0.0, 1.0]$.
