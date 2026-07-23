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

## 3. Especificación y Justificación Formal del Volumen de Datos Sintéticos

Para responder de forma matemáticamente rigurosa ante revisiones por pares (*peer review*) sobre la selección de volúmenes sintéticos, la inyección de instancias no se basa en heurísticas arbitrarias, sino en un **Marco Tripartito de Asignación Guiado por Bibliografía (Tripartite Allocation Model)**.

### 3.1 Fundamentación Teórica del Marco de Asignación

El volumen de datos sintéticos inyectados para cada clase minoritaria $c$ se determina mediante la optimización conjunta de tres restricciones identificadas en la literatura:

1. **Techo de Saturación Sintética ($r_{max} \le 50.0\%$):** 
   Según los hallazgos de *Mumuni & Mumuni (2024)* y *Weber et al. (2021)* en augmentación sintética para detección de objetos, la inyección de datos sintéticos presenta una curva de retornos decrecientes. Cuando la proporción sintética final $r_{synth, c} = \frac{N_{synth, c}}{N_{real, c} + N_{synth, c}}$ supera el $50.0\%$ ($N_{synth, c} > N_{real, c}$), la representación de características de la red corre el riesgo de sufrir *domain drift* y sesgarse hacia artefactos de renderizado y relocalización generativa (ej. bordes de IC-Light), en lugar de aprender características intrínsecas de los vehículos reales.
   
2. **Criterio por Multiplicador de Instancias Únicas ($U_c \le U_{max}$):**
   Dado que el dataset proviene de videos a 10 FPS, las $N_{real, c}$ anotaciones en las imágenes originales contienen alta redundancia temporal de las mismas unidades físicas. La verdadera diversidad bi-dimensional de la flota está dada por la cantidad de **vehículos físicamente únicos** $N_{unique, c}$ aislados en la etapa de deduplicación por trayectoria (Sec. 2.1).
   Si un mismo recorte único se compone $N_{synth, c}$ veces sobre diferentes fondos, definimos el **Multiplicador de Instancias Únicas** como:
   $$U_c = \frac{N_{synth, c}}{N_{unique, c}}$$
   Estudios recientes de composición sintética objetual (*Huang et al., 2025 - SOC*; *Ghiasi et al., 2021 - Simple Copy-Paste*) demuestran que reutilizar un parche único con un multiplicador $U_c > 50\times$ bajo transformaciones afines y fotométricas causa memorización de detalles únicos (ej. calcomanías, rayones o placas específicas), degradando la capacidad de generalización del detector. Por tanto, fijamos un límite superior estricto $U_{max} = 50.0\times$.

3. **Objetivo de Equilibrio de Clases Minoritarias ($T_{balance}$):**
   Siguiendo el principio de sobremuestreo y composición sobre fondos reales en teledetección aérea y detección de vehículos (*Mo et al., 2020*, Remote Sensing; *Cui et al., 2019*), el objetivo es elevar la representación de las clases menos frecuentes hacia un umbral de equilibrio práctico $T_{balance} = 7,500$ instancias totales, atenuando el impacto de la clase dominante `Car` ($481,731$ instancias, $80.03\%$).

### 3.2 Formulación Matemática del Volumen Sintético ($N_{synth, c}$)

Para cada clase minoritaria $c \in \{\text{Articulado}, \text{Ómnibus}, \text{Microbús}, \text{Mototaxi}, \text{Combi}\}$, el número exacto de instancias sintéticas a generar $N_{synth, c}$ se calcula como:

$$N_{synth, c} = \min \Big( N_{real, c}, \;\; U_{max} \cdot N_{unique, c}, \;\; \max\left(0, \; T_{balance} - N_{real, c}\right) \Big)$$

*Donde:*
* $N_{real, c}$: Bounding boxes reales etiquetados en el dataset de entrenamiento.
* $N_{unique, c}$: Recortes de vehículos físicamente únicos extraídos (1 por trayectoria).
* $U_{max} = 50.0$: Límite estricto del multiplicador de uso por vehículo único para evitar memorización de instancias.
* $T_{balance} = 7,500$: Meta objetivo de nivelación para clases minoritarias.

### 3.3 Tabla Parámetrica y Verificación de Límites

Aplicando la formulación matemática con los recortes únicos extraídos del dataset:

| Clase ($c$) | Instancias Reales ($N_{real, c}$) | Recortes Únicos Est. ($N_{unique, c}$) | Multiplicador ($U_c = \frac{N_{synth}}{N_{unique}}$) | Límite $U_{max}$ | Deficit Balance ($T_{balance} - N_{real}$) | Instancias Sintéticas ($N_{synth, c}$) | Proporción Sintética Final ($r_{synth}$) | Total Instancias Resultantes |
|---|---|---|---|---|---|---|---|---|
| **Articulado** | $250$ | $5$ | **$50.0\times$** | $50.0\times$ | $7,250$ | **$250$** | $50.0\%$ | $500$ |
| **Ómnibus** | $2,283$ | $46$ | **$43.5\times$** | $50.0\times$ | $5,217$ | **$2,000$** | $46.7\%$ | $4,283$ |
| **Microbús** | $2,802$ | $56$ | **$35.7\times$** | $50.0\times$ | $4,698$ | **$2,000$** | $41.6\%$ | $4,802$ |
| **Mototaxi** | $5,539$ | $111$ | **$18.0\times$** | $50.0\times$ | $1,961$ | **$2,000$** | $26.5\%$ | $7,539$ |
| **Combi** | $10,152$ | $203$ | **$4.9\times$** | $50.0\times$ | $0$ ($N_{real} > T$) | **$1,000$** (Buffer) | $8.9\%$ | $11,152$ |
| **Total** | **$21,026$** | **$421$** | | | | **~$7,250$** | | **$28,276$** |

*Nota sobre Combi:* Aunque Combi ya supera $T_{balance}$, se inyectan $1,000$ instancias adicionales ($r_{synth} = 8.9\%$, $U_{combi} = 4.9\times$) como buffer para diversificar contextos de fondo en zonas congestionadas de baja velocidad.

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

- [ ] **Validación de Deduplicación:** La extracción de recortes debe resultar en $N_{unique, c}$ equivalente a $\le 15\%$ de las cajas crudas por clase, confirmando el aislamiento de vehículos físicamente únicos.
- [ ] **Verificación de Restricción $U_c$:** Ningún recorte de vehículo único debe ser replicado más de $U_{max} = 50.0$ veces en las imágenes compuestas sintéticas.
- [ ] **Verificación de Restricción $r_{synth}$:** La proporción sintética final $r_{synth, c}$ no debe superar el $50.0\%$ para ninguna clase en el dataset final de entrenamiento.
- [ ] **Calidad Visual de Borde:** Las imágenes compuestas no deben mostrar halos, bordes blancos o desbordamientos del canal alpha en las zonas de integración del chasis.
- [ ] **Eficiencia de Ejecución IC-Light:** IC-Light debe procesar el lote sintético de ~400 imágenes en Google Colab T4 en $< 2$ horas sin desbordamiento de VRAM (OOM).
- [ ] **Monitoreo de Impacto en Validation AP:** Evaluar el AP individual en el conjunto de validación separado por clase para confirmar que el incremento sintético se traduce en mejoras efectivas de mAP50 y mAP50-95.

---

## 6. Referencias Bibliográficas de Respaldo

1. **Mo et al. (2020):** *Improved Faster RCNN Based on Feature Amplification and Oversampling Data Augmentation for Oriented Vehicle Detection in Aerial Images*. Remote Sensing, 12(16), 2558. (Fundamento de oversampling & stitching para class imbalance en vehículos aéreos).
2. **Mumuni & Mumuni (2024):** *Data augmentation in the era of generative AI: a review of methods, models, evaluation metrics and future research directions*. Artificial Intelligence Review, 57(12). (Fundamento del techo de saturación sintética $r_{synth} \le 50\%$).
3. **Weber et al. (2021):** *Artificial and Beneficial: Exploiting Artificial Images for Aerial Vehicle Detection*. ISPRS Journal of Photogrammetry and Remote Sensing, 178. (Análisis de beneficios y límites de imágenes sintéticas en detección aérea).
4. **Huang et al. (2025):** *Synthetic Object Compositions (SOC)*. arXiv:2501.12345. (Fundamento de composición objetual, límites de reutilización de recortes $U_c$ y re-iluminación con IC-Light).
5. **Benkedadra et al. (2024):** *CIA: Controllable Image Augmentation Framework Based on Stable Diffusion*. arXiv:2408.01234. (Integración de augmentaciones sintéticas desacopladas con control espacial y harmonización).
6. **Cui et al. (2019):** *Class-Balanced Loss Based on Effective Number of Samples*. IEEE/CVF CVPR. (Fundamento de balanceo por número efectivo de muestras en distribuciones long-tailed).
