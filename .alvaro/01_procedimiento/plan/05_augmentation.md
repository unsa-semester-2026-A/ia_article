# Fase 2: Aumentación Generativa de Clases Minoritarias (05_augmentation.md)

Este documento detalla las especificaciones de diseño, flujo de trabajo, control de calidad y justificaciones para la inyección de datos sintéticos armonizados de las clases menos representadas del dataset.

---

## 1. Objetivo
Mitigar el sesgo introducido por el desbalance extremo de clases (ej. ómnibus articulados representando el 0.04% del dataset frente al 80.03% de autos) inyectando instancias sintéticas realistas. Estas instancias se componen sobre fondos reales limpios y se re-iluminan con **IC-Light** para evitar la memorización de instancias únicas y cerrar el *domain gap*.

---

## 2. Especificación de los Módulos de Aumentación

### 2.1 Módulo: Extracción y Deduplicación de Recortes (Crops)
El objetivo de este módulo es aislar parches visuales de las clases minoritarias de las imágenes originales.

* **Entrada:** Imágenes reales, DataFrame de anotaciones parseadas.
* **Proceso de Extracción:**
  1. Para cada caja de Ground Truth perteneciente a las clases minoritarias (`Combi`, `Microbus`, `Minibus`, `Omnibus`, `Articulated bus`, `Mototaxi`):
     - Rotar la imagen original en dirección contraria al ángulo anotado ($- \theta_{deg}$) tomando el centro $(cx, cy)$ como pivote, alineando el vehículo horizontalmente.
     - Recortar la caja axis-aligned correspondiente a las dimensiones $(w, h)$ añadiendo un margen de seguridad de $10\%$ para capturar los bordes.
     - Generar un mapa de transparencia (canal alpha) calculando la elipse inscrita en el rectángulo del vehículo. Esto separa el chasis del fondo original de la pista.
     - Guardar el recorte como un archivo PNG con transparencia (RGBA).
* **Deduplicación Temporal (CRÍTICO):**
  - Dado que los videos corren a 10 FPS, una misma combi o mototaxi aparece en ~50 frames continuos del mismo clip con sutiles variaciones. Guardar todos los frames causaría una redundancia extrema.
  - **Regla de Selección:** Extraer únicamente **1 recorte por vehículo único por clip** (seleccionando el frame medio de la trayectoria del vehículo).
  - Esto reduce las instancias a vehículos visualmente únicos en la flota real.

---

### 2.2 Módulo: Composición en Posiciones Realistas
El objetivo de este módulo es colocar los recortes en zonas coherentes del asfalto para mantener el contexto físico.

* **Estrategia de Posicionamiento:** **¿Dónde pegamos el vehículo?**
  - Para evitar colocar un camión flotando en un techo o una combi perpendicular a los carriles, utilizaremos la lista de posiciones del archivo JSON generado por la Fase 0 (pseudo-labeling).
  - Pega el recorte sintético **exactamente en las coordenadas $(cx, cy, \theta)$ de un vehículo estacionado que LaMa borró previamente** en esa misma imagen.
  - Esto garantiza que el vehículo sintético heredará una escala y una orientación vial físicamente coherentes para esa intersección específica.

* **Mitigación de la Memorización de Instancia (Google Colab):**
  - Contamos con muy pocas instancias únicas reales (ej. ~5 buses articulados únicos). El modelo final podría memorizar estos 5 vehículos de forma exacta.
  - **Transformaciones de Variabilidad:** Antes de pegar el recorte sobre el fondo limpio, se le aplican transformaciones aleatorias de apariencia:
    - *Color Jitter:* Brillo aleatorio en $\pm 20\%$, contraste en $\pm 15\%$, saturación en $\pm 20\%$.
    - *Ruido Gaussiano:* Agregar ruido sutil ($\sigma = 5$ a $15$) para simular variaciones del sensor de la cámara.
    - *Escala:* Factor aleatorio de $0.8\times$ a $1.2\times$ respecto a la escala original del parche.
    - *Flip Horizontal:* $50\%$ de probabilidad (invierte la dirección del vehículo).
    - *Rotación sutil:* $\pm 10^\circ$ sobre el ángulo original de la pista.

---

### 2.3 Módulo: Armonización y Re-iluminación (IC-Light)
El objetivo de este módulo es ajustar el color, el balance de luces y generar sombras coherentes en el vehículo compuesto para que se integre de forma fotorrealista con la iluminación del fondo.

* **Herramienta:** `IC-Light` (Foreground-Background Conditioned: `iclight_sd15_fbc.safetensors`).
* **Entorno de Ejecución:** Google Colab T4. El modelo requiere ~6 GB de VRAM a una resolución de 512×512 píxeles. Inferencia de 25 pasos por imagen.
* **Proceso de Armonización:**
  1. Redimensionar la imagen compuesta y la máscara binaria del vehículo a $512\times512$.
  2. Ejecutar IC-Light enviando el prompt contextual: `"outdoor urban road, daylight, traffic, aerial view"`.
  3. IC-Light regenera el color y sombreado del vehículo para acoplarlo al fondo, proyectando sombras coherentes en la pista limpia de LaMa.
  4. Redimensionar la imagen armonizada de vuelta a su resolución original mediante interpolación Lanczos.
  5. Guardar la imagen final y generar el archivo de anotaciones YOLO-OBB que incluye el nuevo objeto sintético agregado.

---

## 3. Especificación del Volumen de Datos Sintéticos a Generar

El volumen de datos sintéticos inyectados sigue un criterio estricto: **no superar el 50% de la proporción de instancias totales de esa clase** para no saturar al modelo con datos sintéticos repetitivos de los mismos pocos vehículos base.

| Clase | Instancias Reales | Sintéticas a Generar | Proporción Sintética Final | Total Instancias |
|---|---|---|---|---|
| **Articulado** | $250$ | **$250$** (1× oversampling) | $50.0\%$ | $500$ |
| **Ómnibus** | $2,283$ | **$2,000$** | $46.7\%$ | $4,283$ |
| **Microbús** | $2,802$ | **$2,000$** | $41.6\%$ | $4,802$ |
| **Mototaxi** | $5,539$ | **$2,000$** | $26.5\%$ | $7,539$ |
| **Combi** | $10,152$ | **$1,000$** | $8.9\%$ | $11,152$ |
| **Total** | | **~7,250 instancias** | | |

* **Imágenes compuestas estimadas:** ~400 imágenes. Cada imagen sintética de entrenamiento contendrá entre 3 y 5 vehículos minoritarios pegados en paralelo sobre el fondo limpio.

---

## 4. Estructura del Dataset Sintético Resultante

```
synthetic_augmented/
├── images/
│   ├── v_synth_0000.jpg (Fondos LaMa con vehículos sintéticos re-iluminados)
│   └── ... (~400 imágenes)
└── labels/
    ├── v_synth_0000.txt (labels que anotan TODOS los objetos: reales y sintéticos)
    └── ...
```

---

## 5. Criterios de Aceptación
- [ ] La deduplicación de recortes debe resultar en menos del 15% del total de cajas crudas (indicando que se aislaron vehículos únicos).
- [ ] Las imágenes compuestas no deben mostrar desbordamientos visuales o bordes blancos alrededor del vehículo compuesto (verificar canal alpha).
- [ ] IC-Light debe procesar el lote completo en Colab en menos de 2 horas sin errores de memoria (OOM).
- [ ] El AP final del conjunto de validación debe ser monitoreado individualmente para verificar si las clases aumentadas muestran mejoras significativas.
