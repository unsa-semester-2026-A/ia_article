# [DESCARTADO] Fase 2: Aumentación Generativa de Clases Minoritarias

> [!WARNING]
> **Documento descartado.** La Fase 2 con IC-Light fue retirada del alcance del estudio y
> reemplazada por la comparativa entre familias de arquitecturas de detección orientada
> especificada en `05_architecture_comparison.md`.
>
> Se conserva por dos motivos: (1) la fundamentación matemática del volumen sintético de la
> Sección 3 y sus citas (Mumuni & Mumuni 2024, Benkedadra et al. 2024, Mo et al. 2020) siguen
> siendo material válido para la sección de *Trabajo Futuro* del artículo; (2) el módulo de
> extracción y deduplicación de recortes de la Sección 2.1 documenta el conteo de vehículos
> físicamente únicos por clase, dato que se cita en el análisis de desbalance.

Este documento detalla las especificaciones de diseño, flujo de trabajo, control de calidad y justificaciones para la inyección de datos sintéticos armonizados de las clases menos representadas del dataset.

---

## 1. Objetivo

Mitigar el sesgo introducido por el desbalance extremo de clases inyectando instancias sintéticas realistas. Estas instancias se componen sobre fondos reales limpios y se re-iluminan generativamente. El objetivo es evitar la memorización de instancias únicas, mantener una alineación estricta de las cajas delimitadoras para no degradar el rendimiento del modelo, y cerrar la brecha de dominio (domain gap).

---

## 2. Especificación de los Módulos de Aumentación

### 2.1 Módulo: Extracción y Deduplicación de Recortes (Crops)

El objetivo de este módulo es aislar parches visuales de las clases minoritarias de las imágenes originales.

**Entrada:** Imágenes reales, DataFrame de anotaciones parseadas.

**Proceso de Extracción:**

1. Para cada caja de Ground Truth perteneciente a las clases minoritarias, rotar la imagen original en dirección contraria al ángulo anotado ($- \theta_{deg}$) tomando el centro $(cx, cy)$ como pivote, alineando el vehículo horizontalmente.
2. Recortar la caja correspondiente a las dimensiones $(w, h)$ añadiendo un margen de seguridad de **10%** para capturar los bordes.
3. Generar un mapa de transparencia (canal alpha) calculando la elipse inscrita en el rectángulo del vehículo para separar el chasis del fondo original de la pista.
4. Guardar el recorte como un archivo PNG con transparencia (RGBA).

**Deduplicación Temporal (CRÍTICO):**
Dado que los videos corren a 10 FPS, guardar todos los frames causaría redundancia. Se debe extraer únicamente **1 recorte por vehículo único por clip** seleccionando el frame medio de la trayectoria. Esto reduce las instancias a vehículos físicamente únicos en la flota real.

---

### 2.2 Módulo: Composición en Posiciones Realistas

El objetivo de este módulo es aplicar sobremuestreo y pegado (oversampling and stitching) sobre imágenes de fondo para mitigar el desbalance de clases.

**Estrategia de Posicionamiento:**
Pegar el recorte sintético exactamente en las coordenadas $(cx, cy, \theta)$ de un vehículo que fue borrado previamente en esa misma imagen. Esto garantiza heredar una escala y orientación vial físicamente coherentes.

**Mitigación de Memorización en Tiempo de Generación:**
Antes de pegar el recorte sobre el fondo limpio, se aplican transformaciones aleatorias de apariencia:

- **Color Jitter:** Brillo aleatorio en **± 20%**, contraste en **± 15%**, saturación en **± 20%**.
- **Ruido Gaussiano:** Agregar ruido sutil ($\sigma = 5$ a $15$).
- **Escala:** Factor aleatorio de $0.8\times$ a $1.2\times$.
- **Flip Horizontal:** Probabilidad del **50%**.
- **Rotación Sutil:** Diferencial de $\pm 10^\circ$ sobre el ángulo original de la pista.

---

### 2.3 Módulo: Armonización y Re-iluminación (IC-Light)

El objetivo es ajustar el color y sombreado para fotorrealismo sin alterar el área visual del objeto fuera de su caja delimitadora original.

**Herramienta y Entorno:**

- Uso de `IC-Light` condicionado a primer y segundo plano.
- Ejecución en Google Colab T4. Requiere ~6 GB de VRAM a resolución de $512\times512$.

**Proceso de Armonización:**

1. Aplicar _letterboxing_ (relleno con bordes) y redimensionar la imagen compuesta a $512\times512$ para no deformar el aspecto de vehículos alargados.
2. Ejecutar IC-Light con el prompt contextual de tráfico urbano.
3. Redimensionar la imagen armonizada de vuelta a su resolución y relación de aspecto original mediante interpolación Lanczos.
4. Guardar la imagen final y actualizar el archivo YOLO-OBB.

---

## 3. Especificación y Justificación Formal del Volumen de Datos Sintéticos

La inyección de instancias se rige por límites matemáticos estrictos basados en evidencias sobre saturación, brecha de dominio y sobreajuste.

### 3.1 Fundamentación Teórica del Marco de Asignación

El volumen sintético se optimiza mediante tres restricciones:

**1. Techo Híbrido Local y Evasión de Saturación Global ($r_{max} \le 50.0\%$):**

- La evidencia empírica (Richter et al. y Zhang et al., citados en Mumuni & Mumuni, 2024) demuestra que depender exclusivamente de datos sintéticos reduce drásticamente el rendimiento, y que las mejoras se estancan cuando los datos sintéticos alcanzan aproximadamente el 25% de la composición total del dataset de entrenamiento.
- Dado que el dataset cuenta con una clase mayoritaria `Car` de 481,731 instancias, la inyección sintética proyectada (~7,200 instancias) representará menos del 1.5% del volumen global del dataset, evadiendo por completo la saturación general del 25%.
- No obstante, para evitar el _domain gap_ a nivel de **clase individual**, se impone la restricción $N_{synth, c} \le N_{real, c}$. Esto garantiza que la proporción sintética final $r_{synth, c}$ nunca supere el **50.0%** dentro de una misma categoría minoritaria, asegurando que los enfoques híbridos prioricen anclar el aprendizaje en las características del mundo real.

**2. Factor de Reutilización de Instancias Únicas ($U_c \le U_{max}$):**

- Los métodos clásicos de aumentación son propensos a causar sobreajuste (overfitting) si no existe una correcta selección y control (Benkedadra et al., 2024).
- Para formalizar la variabilidad real frente a la sintética, definimos el multiplicador $U_c = \frac{N_{synth, c}}{N_{unique, c}}$.
- Acotamos empíricamente este multiplicador a $U_{max} = 50.0\times$ como techo máximo absoluto para evitar que la red neuronal memorice texturas de alta frecuencia de los escasos recortes únicos extraídos.

**3. Meta de Expansión Empírica ($T_{balance}$):**

- El sobremuestreo y composición sobre imágenes de fondo permite sintetizar un nuevo dataset equilibrado (Mo et al., 2020).
- Se establece un _expansion benchmark_ (punto de referencia de expansión) empírico de $T_{balance} = 7,500$ instancias.
- Esta meta detiene la inyección generativa una vez que las clases minoritarias han igualado el orden de magnitud básico necesario para no ser ignoradas por la función de pérdida del detector.

### 3.2 Formulación Matemática del Volumen Sintético ($N_{synth, c}$)

Para cada clase minoritaria $c$, el volumen máximo seguro de instancias sintéticas a generar ($N_{synth, c}$) se calcula mediante la siguiente optimización con restricciones:

$$N_{synth, c} = \min \Big( N_{real, c}, \;\; U_{max} \cdot N_{unique, c}, \;\; \max\left(0, \; T_{balance} - N_{real, c}\right) \Big)$$

_(Nota: Esta fórmula dictamina el límite superior seguro de inyección. Operativamente, las cuotas finales pueden redondearse o ajustarse por debajo de este límite dictaminado)._

### 3.3 Tabla Parámetrica y Verificación de Límites

| Clase ($c$)    | Instancias Reales ($N_{real, c}$) | Recortes Únicos Est. ($N_{unique, c}$) | Límite por Memorización ($U_{max} \cdot N_{unique, c}$) | Límite por Balance ($T_{balance} - N_{real}$) | Cuota Sintética Asignada | Proporción Sintética Resultante |
| -------------- | --------------------------------- | -------------------------------------- | ------------------------------------------------------- | --------------------------------------------- | ------------------------ | ------------------------------- |
| **Articulado** | 250                               | 5                                      | 250                                                     | 7,250                                         | **250**                  | 50.0%                           |
| **Ómnibus**    | 2,283                             | 46                                     | 2,300                                                   | 5,217                                         | **2,000**                | 46.7%                           |
| **Microbús**   | 2,802                             | 56                                     | 2,800                                                   | 4,698                                         | **2,000**                | 41.6%                           |
| **Mototaxi**   | 5,539                             | 111                                    | 5,550                                                   | 1,961                                         | **1,961**                | 26.1%                           |
| **Combi**      | 10,152                            | 203                                    | 10,150                                                  | 0 (Superado)                                  | **1,000** (Buffer)       | 8.9%                            |

_(El "buffer" de 1,000 para Combis se inyecta estrictamente bajo el techo híbrido del 50% y el límite de reutilización, con el único fin de incrementar la variabilidad de fondos)._

---

## 4. Estructura del Dataset Sintético Resultante

```text
synthetic_augmented/
├── images/
│   ├── v_synth_0000.jpg (Fondos LaMa con vehículos sintéticos re-iluminados)
│   └── ...
└── labels/
    ├── v_synth_0000.txt (Anotaciones conjuntas reales y sintéticas)
    └── ...

```

---

## 5. Criterios de Aceptación

- **Entrenamiento Híbrido Obligatorio:** El entrenamiento final utilizará la unión del conjunto de datos original y el sintético generado; sustituir datos reales por sintéticos degrada el rendimiento de los detectores.
- **Validación de Deduplicación:** La extracción de recortes aísla físicamente los vehículos únicos comprobando que $N_{unique, c}$ representa $\le 15\%$ de las cajas crudas originales.
- **Muestreo Aleatorio:** La selección final de imágenes a integrar al entrenamiento se realizará mediante muestreo aleatorio estándar, ya que priorizar imágenes evaluando métricas complejas de calidad visual (IQA) no supera en eficacia a un muestreo estocástico puro.

- **Calidad Geométrica:** Las imágenes compuestas no alterarán la geometría del objeto original (sin desbordamientos del canal alpha o distorsiones de relación de aspecto), manteniendo coherencia total con la caja anotada.
- **Monitoreo de Impacto:** Evaluar el rendimiento general separando el conjunto de validación (conformado exclusivamente por datos reales) para medir el incremento del mAP50 y mAP50-95.
