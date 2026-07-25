# Mapa de Riesgos y Contingencias (08_risks.md)

Este documento detalla la matriz de riesgos del proyecto, las acciones preventivas, las acciones
correctivas y los criterios de priorización si el tiempo o los recursos se agotan.

> [!IMPORTANT]
> El deadline original del 22 de julio de 2026 ya venció. **Falta fijar la nueva fecha de entrega**;
> las prioridades de la Sección 3 son válidas con cualquier fecha, pero el margen de holgura no se
> puede calcular hasta anclarla.

---

## 1. Matriz de Riesgos

### 1.1 Riesgos del eje arquitectónico (nuevos)

| ID | Evento de riesgo | Prob. | Impacto | Mitigación | Contingencia |
|---|---|---|---|---|---|
| **R8** | **`mmcv.ops` no compila o no importa en Kaggle**, bloqueando F2 y F3 por completo | Alta | Crítico | Spike bloqueante el Día 1 antes de comprometer cualquier cuota (`10_environment_mmrotate.md` §3.1). Construir las wheels una sola vez en Colab y publicarlas como Kaggle Dataset para no recompilar nunca. | Escalera de 4 niveles de `10_environment_mmrotate.md` §5: (2) entorno legacy con `torch==1.13` + `mmcv-full==1.7.2`; (3) F2 con `oriented-det` en PyTorch puro y F3 con mmrotate, declarando frameworks mixtos; (4) sustituir F2 por Rotated Faster R-CNN. |
| **R9** | **Error silencioso de convención angular** (le90 vs grados) que baja el AP de F2/F3 sin lanzar excepción | Alta | Crítico | Test de equivalencia obligatorio con rIoU ≥ 0.999 sobre 100 OBB aleatorios, incluyendo ángulos negativos, cercanos a ±90° y cajas con $w<h$. El test debe correr **antes** de evaluar cualquier modelo. | Si un AP de F2/F3 sale anómalamente bajo, la primera hipótesis es siempre el adaptador y no el modelo: verificar visualmente 20 OBB predichos dibujados sobre la imagen antes de gastar cuota reentrenando. |
| **R10** | **Compilar `mmcv` consume la cuota de GPU** en cada sesión (40-90 min por reinicio) | Alta | Alto | Kaggle Dataset `mmrotate-wheels` con instalación offline en < 5 min. Construir las wheels en Colab, que no gasta cuota de Kaggle. | Construir las wheels en la VM local si Colab falla, aunque la arquitectura de CUDA deba coincidir. |
| **R11** | **Divergencia de versiones entre C1 y C3 de una misma familia**, que invalida el $\Delta_F$ (resultado principal) | Media | Crítico | `experiments/requirements-mmrotate.lock` versionado más el Kaggle Dataset de wheels, que hace el entorno reproducible bit a bit en cualquier cuenta. La verificación es un diff del `pip freeze` volcado por C1 y C3 (`06_training.md` §6.2). | Descartar el par y reentrenar ambas condiciones con el entorno congelado. Reportar un $\Delta_F$ con entornos distintos no es aceptable. |
| **R12** | **OOM en los modelos R50-FPN** con rotated RoIAlign a 640 px | Media | Alto | `batch_size=4` por GPU, conservador de antemano precisamente por esto. Corrida de humo obligatoria de 200 iteraciones que mide el pico de VRAM antes de comprometer horas de cuota (`11_training_runbook.md` S7). | Reducir a `batch_size=2` **en ambas condiciones de la familia**, nunca en una sola. Como último recurso, congelar `stem` + `layer1` del backbone, también en ambas condiciones. |
| **R13** | **F2/F3 no disparan el early stopping** y una corrida se va hacia las 32 h del tope de 40 épocas, consumiendo la cuota de una cuenta completa | Media | Alto | `min_delta=0.001` en el `EarlyStoppingHook`, que impide que una mejora del cuarto decimal reinicie el contador de paciencia. Monitoreo de la corrida en las primeras épocas. | Tope operativo de 24 épocas (`06_training.md` §6.1), **aplicado simétricamente a C1 y C3** de la familia y reportado como tope y no como convergencia. Un tope asimétrico destruiría $\Delta_F$. |
| **R15** | **El reescalado de las imágenes de LaMa usa un filtro de interpolación distinto al de `train_resized`**, introduciendo una diferencia sistemática de nitidez entre C1 y C3 que se confunde con el efecto de la limpieza | Media | **Crítico** | Verificación explícita en `11_training_runbook.md` S0. Si no consta qué filtro se usó para `train_resized`, regenerar **ambas** variantes a 640×360 en la misma corrida y con el mismo código. | Si se detecta después de entrenar, las corridas C3 afectadas son inservibles y hay que repetirlas: el $\Delta_F$ mediría nitidez y no limpieza de etiquetas. Es el riesgo más barato de prevenir y el más caro de descubrir tarde. |
| **R16** | **C1 y C3 no ven el mismo número de imágenes** porque LaMa solo se aplicó a los frames con autos estacionados detectados | Media | Alto | Rellenar la variante LaMa con copias sin modificar de los frames no tocados, y verificar la igualdad de los conjuntos de nombres de archivo en S0. | Corregir el dataset y repetir C3. Un $\Delta_F$ con volúmenes de datos distintos mezcla dos efectos y no es reportable. |
| **R14** | **Los pesos DOTA de mmrotate no están disponibles** cuando se necesitan | Baja | Medio | Descargar y respaldar los checkpoints en Google Drive el Día 1, junto con las wheels. | Entrenar F2/F3 desde pesos de ImageNet, declarando que su Base 0 no existe y que su preentrenamiento es más débil que el de F1. |

### 1.2 Riesgos del pipeline de datos (heredados)

| ID | Evento de riesgo | Prob. | Impacto | Mitigación | Contingencia |
|---|---|---|---|---|---|
| **R1** | El pseudo-labeler temporal tiene baja precisión (<90 %) | Media | Alto | Ajustar umbrales de detección y número mínimo de frames sobre un subconjunto de prueba en Colab antes de correr todo el dataset. | Revisión manual más estricta, o limitar el inpainting a los 200 clips con mayor tasa de error visible. |
| **R2** | LaMa genera artefactos visuales notorios | Baja | Medio | Dilatación conservadora de máscara (10 px) y auditoría visual de 100 imágenes. | **Fallback de máscara simple:** sustituir la región por un rectángulo negro o ruido gaussiano. Sirve además como baseline de control que valida si la reconstrucción fotorrealista es indispensable. |
| **R5** | La cuota de GPU de Kaggle (30 h/cuenta) se agota | Baja | Alto | Las 7 corridas suman ~45-55 h esperadas sobre una capacidad de 150 h, con más del 100 % de holgura, gracias a la paciencia de 5 épocas calibrada con el piloto (`06_training.md` §6). Ninguna cuenta carga más de ~16 h. Precedencia de ejecución que sacrifica primero lo prescindible. | Usar Colab Free con corridas reanudadas desde checkpoint, o el modo "Save Version → Run All in Background" de Kaggle (hasta 12 h por ejecución). Aplicar la reducción de alcance de la Sección 3. |
| **R6** | El filtro de movimiento es demasiado estricto o demasiado laxo | Media | Alto | Calibrar sobre el subconjunto de entrenamiento, donde los vehículos estacionados son conocidos. | Auditar visualmente el recall de 20 clips de validación y ajustar `motion_threshold` en pasos de ±2 px. El umbral final es **el mismo para las tres familias**, sin excepción. |
| **R7** | No se llega a redactar el artículo a tiempo | Media | Alto | Redactar Introducción, Metodología y Related Work desde el Día 2, en paralelo con los entrenamientos. La sección de Related Work del eje arquitectónico se puede escribir por completo antes de tener un solo resultado. | Aplicar la reducción de alcance de la Sección 3. |

**Riesgos retirados:** R3 (OOM de IC-Light) y R4 (memorización por oversampling generativo) dejan de
aplicar al descartarse la Fase 2 con IC-Light.

---

## 2. Riesgo Metodológico Principal

El punto más atacable del artículo no es técnico sino de diseño: **comparar detectores entrenados en
frameworks distintos**. Un revisor puede objetar que la diferencia medida entre F1 y F2 se debe al
backbone (11 M vs 41 M parámetros), al optimizador (AdamW vs SGD) o al preentrenamiento, y no al
paradigma arquitectónico.

La defensa está integrada en el diseño y debe explicitarse en la sección de Metodología:

1. **El resultado principal es intra-familia.** Cada $\Delta_F$ compara una familia contra sí misma
   con backbone, optimizador, schedule, semilla y framework idénticos. Es estructuralmente inmune a
   esta objeción, y por eso es el resultado que se destaca, no el ranking absoluto entre familias.
2. **Toda tabla que compare familias entre sí lleva columnas de parámetros, GFLOPs y latencia**, de
   modo que ninguna lectura pueda asumir igualdad de capacidad.
3. **Las diferencias se declaran, no se ocultan** (`05_architecture_comparison.md` §5.2), incluida
   la asimetría de medición de latencia entre ONNX y PyTorch, que además favorece a F1 y por tanto
   no puede sesgar H3 en la dirección que se quiere demostrar.
4. **La corrida opcional de `YOLO26m-obb`** acota cuánto del gap absoluto es capacidad y cuánto es
   paradigma, si la cuota lo permite.

---

## 3. Priorización de Experimentos (Ablación Mínima Viable)

Se aplica en orden secuencial si el tiempo o la cuota se agotan.

### Nivel 1: Alcance completo (7 corridas + 3 zero-shot)

B0 ×3, C1 ×3, C2, C3 ×3. Requiere el entorno de mmrotate funcionando y las 5 cuentas activas. La
corrida E1 de control de escala del plan anterior ya no existe: desapareció junto con el submuestreo
temporal que existía para cuantificar (`06_training.md` §2.1).

### Nivel 2: Eliminar las 3 evaluaciones zero-shot (B0)

Ahorra ~2 h de inferencia y todo el trabajo de mapeo de clases DOTA→SMART. Costo bajo: B0 es la
condición menos informativa y su resultado es predecible (AP muy bajo, con dos clases en cero por
construcción).

### Nivel 3: Eliminar C2 (aumento clásico en F1)

Ahorra ~4 h. Costo: se pierde la respuesta a "¿el aumento clásico logra lo mismo que limpiar?", que
es una pregunta secundaria frente a la universalidad del efecto.

### Nivel 4: Ablación Mínima Viable (6 corridas)

**C1 y C3 en las tres familias.** Este es el núcleo indispensable y **no es reducible**: los tres
pares $\Delta_F$ son literalmente la contribución del artículo. Si no caben seis corridas, el
estudio no puede sostener su tesis.

### Nivel 5: Degradación de emergencia (4 corridas)

Si el entorno de mmrotate falla de forma irrecuperable en los cuatro niveles de
`10_environment_mmrotate.md` §5, reducir a dos familias: **F1 (YOLO26s) y F2 (Oriented R-CNN vía
`oriented-det`, que no necesita MMCV)**, con C1 y C3 en cada una. Se conserva el contraste más
importante —negativos densos vs negativos muestreados, que es el que sustenta H2— y se reencuadra el
artículo como comparación entre dos paradigmas en lugar de tres. F3 pasa a Trabajo Futuro.

---

## 4. Contingencia del Pipeline de Datos

Si la cuota de Drive falla o los tiempos de transferencia en Colab son excesivos:

1. **Descarga directa en Kaggle:** bajar el dataset dentro de cada notebook con la API de la
   competencia (>100 MB/s de red interna de Google).
2. **Procesamiento en Kaggle:** ejecutar LaMa dentro de un notebook de Kaggle, guardar en
   `/kaggle/working/` y exportar como dataset privado, evitando mover 40 GB por equipos locales.
3. **Conversión a DOTA en Kaggle:** el conversor es puro CPU y rápido, así que puede ejecutarse al
   inicio de cada notebook de F2/F3 sobre el dataset ya montado, evitando mantener y transferir una
   segunda copia del dataset en Drive.
