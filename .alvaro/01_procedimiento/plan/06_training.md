# Configuración de Entrenamientos y Ablación (06_training.md)

Este documento detalla las especificaciones de entrenamiento, la asignación de plataformas en la
nube (Kaggle) y la justificación de los hiperparámetros de las condiciones experimentales del
estudio, para las tres familias de detectores orientados definidas en
`05_architecture_comparison.md`.

Los pasos operativos concretos, en orden de ejecución, están en `11_training_runbook.md`. Este
documento define **qué** se entrena y con qué valores; el runbook define **cómo** se lanza.

---

## 1. Objetivo

Entrenar tres detectores de familias arquitectónicas disjuntas bajo condiciones controladas para
aislar dos efectos: (a) el aporte de la limpieza de ruido de etiquetas con LaMa dentro de cada
familia, y (b) la interacción entre ese aporte y el mecanismo de asignación de etiquetas de cada
arquitectura.

El resultado principal es la ganancia **intra-familia**
$\Delta_F = \text{AP}(\text{C3}) - \text{AP}(\text{C1})$, medida con backbone, optimizador,
schedule y framework idénticos dentro de cada familia. Toda decisión de este documento se
subordina a proteger la validez de ese delta.

---

## 2. Evidencia Empírica de la Corrida Piloto

Todas las estimaciones de este documento se apoyan en una corrida real de `YOLO26s-OBB` C1 sobre el
dataset completo, ejecutada en Kaggle con 2× Tesla T4. Las cifras salen de su log, no de
extrapolaciones:

| Magnitud medida | Valor |
|---|---|
| Imágenes de entrenamiento | 43,392 (452 iteraciones × batch 96) |
| Imágenes de validación | 10,873, con 109,448 instancias anotadas |
| Tiempo de entrenamiento por época | **14 min 09 s** (1.88 s/iteración) → **51 img/s** |
| Tiempo de época incluyendo validación completa | **~14.7 min** (~880 s) |
| VRAM por GPU | 13.6 GB de 15 GB con batch 96 (48 por GPU en DDP) |
| Épocas completadas | 39 de 100, en ~9.5 h de cómputo |

### 2.1 Las dos conclusiones que reorientan el plan

**Primera: el modelo converge en la época 6.** El `mAP50-95` interno alcanza 0.726 hacia la época 4
y 0.742 hacia la época 6, y de ahí hasta la época 39 oscila entre 0.73 y 0.75, con un máximo de
0.753. Es decir, **33 épocas adicionales aportaron ~0.01 de mAP**. Con `patience=5` la corrida se
habría detenido cerca de la época 11, en ~2.7 h en lugar de 9.5 h, sacrificando como máximo un
punto de mAP. Esto valida cuantitativamente la recomendación de reducir la paciencia y confirma que
**la paciencia, y no el volumen de datos, es la variable que libera el presupuesto de cómputo**.

**Segunda: el submuestreo temporal del dataset queda descartado.** La versión anterior de este
documento proponía conservar 1 de cada 5 frames para bajar de 43,400 a ~8,700 imágenes. Con early
stopping ese recorte no ahorra tiempo: lo que determina el tiempo total es el número de **muestras
procesadas hasta converger** (~260 k según el piloto), no el tamaño de la época. Submuestrear 5×
solo hace las épocas 5× más cortas, así que se necesitarían ~5× más épocas para procesar las mismas
muestras, y además rompería la calibración de `patience` medida en épocas. El recorte introducía un
riesgo metodológico real (una corrida de control adicional, una objeción previsible del revisor)
sin beneficio de cómputo. **Se entrena con el dataset completo.**

**Tercera, operativa: `cache=True` no funciona en este dataset.** El log registra
`41.9GB RAM required to cache images with 50% safety margin but only 26.5/31.3GB available, not
caching`. Ultralytics degradó silenciosamente a lectura desde disco para el `train` y solo cacheó
el `val` (7.0 GB). Como el throughput de 51 img/s se midió **sin** caché de entrenamiento, el
caché no es necesario y debe fijarse explícitamente a `False` para eliminar el riesgo de OOM y el
minuto de escaneo inútil por sesión.

---

## 3. Dataset Base: `smart-640`

Todas las corridas usan el dataset completo, sin submuestrear.

| Propiedad | Valor |
|---|---|
| Frames de entrenamiento | **43,392** (dataset completo, 10 FPS) |
| Frames de validación | **10,873** (completos, con 109,448 instancias) |
| Resolución de las imágenes en disco | **640×360** (directorio `train_resized/`) |
| Split | Nivel de clip, `seed=42`, 80/20 |
| Variantes | `smart-640` (original) y `smart-640-lama` (limpiado con LaMa) |

### 3.1 Decisión de resolución: 640×360 para las tres familias

Las imágenes originales son de 1920×1080 y existe además una versión reescalada a 640×360
(`train_resized/`, y su equivalente en `smart_lama_corrected/`). **Las tres familias entrenan con
las imágenes de 640×360.** Cuatro razones, en orden de peso:

1. **Comparabilidad.** Los píxeles de entrada son el invariante más fuerte disponible en una
   comparación entre frameworks distintos. Si F2/F3 vieran 1920×1080 y F1 viera 640, ninguna
   afirmación cruzada entre familias sobreviviría a una revisión, y el propio $\Delta_F$ quedaría
   confundido con un efecto de escala.
2. **Factibilidad.** 1920×1080 tiene **9× más píxeles**. Extrapolando desde el piloto, una época de
   F1 pasaría de 14 min a ~2 h, y una de Oriented R-CNN a más de 6 h. El estudio completo sería
   imposible por un factor de ~9.
3. **El techo de rendimiento no está en la resolución.** El piloto alcanza `mAP50-95` ≈ 0.75 a
   640×360. La resolución no es la restricción activa sobre la detectabilidad en este dataset.
4. **El factor de escala es exactamente 3.** $1920/640 = 1080/360 = 3$, así que el reescalado
   preserva la relación de aspecto sin distorsión y las predicciones vuelven a coordenadas
   oficiales de 1920×1080 multiplicando por 3 exacto, sin asimetría de redondeo entre ejes. Este
   factor 3 debe aplicarse en el adaptador de predicciones de `07_evaluation.md` §2 **antes** de
   calcular la métrica, porque el ground truth oficial está en 1920×1080.

**Limitación a declarar en el artículo:** un vehículo de 60 px en el original mide ~20 px a
640×360. El AP absoluto reportado está acotado por este submuestreo espacial, y la comparación es
válida *a resolución fija*. Existe además una amenaza secundaria a la validez que conviene
anticipar: los artefactos de autos estacionados también se reducen 3×, de modo que el efecto de
LaMa medido a 640×360 podría ser una **cota inferior** del efecto a resolución nativa. Esto
favorece la dirección conservadora de la hipótesis (si el efecto aparece a pesar del recorte, es
robusto), y así debe argumentarse.

### 3.2 Geometría de entrada idéntica entre frameworks

Ultralytics con `imgsz=640` aplica letterbox sobre un lienzo de 640×640 con relleno gris. Para que
mmrotate vea exactamente la misma geometría (mismo tamaño de objeto en píxeles y mismo relleno), su
pipeline usa `Resize(scale=(640, 640), keep_ratio=True)` seguido de `Pad(size=(640, 640))`.

Esto no es un detalle cosmético: una diferencia de escala entre frameworks se manifestaría como una
diferencia de AP atribuida erróneamente a la arquitectura. **Criterio de verificación obligatorio:**
volcar un batch preprocesado de cada framework a PNG y confirmar que el ancho en píxeles de un mismo
vehículo coincide dentro de ±1 px. Está en la lista de aceptación de §8.

---

## 4. Matriz de Condiciones Experimentales

| ID | Condición | Dataset | F1 `YOLO26s-OBB` | F2 `Oriented R-CNN` | F3 `S²A-Net` |
|---|---|---|---|---|---|
| **B0** | Zero-shot | — (pesos DOTA) | Solo evaluación | Solo evaluación | Solo evaluación |
| **C1** | Data cruda | `smart-640` | Entrena | Entrena | Entrena |
| **C2** | Aumento clásico | `smart-640` | Entrena | — | — |
| **C3** | Data limpia LaMa | `smart-640-lama` | Entrena | Entrena | Entrena |

**Total: 7 entrenamientos y 3 evaluaciones zero-shot.**

La corrida E1 de "control de escala" del plan anterior **desaparece**: existía únicamente para
cuantificar la pérdida por submuestreo, y sin submuestreo no tiene objeto. Esto libera ~9 h de
cuota y elimina una objeción del revisor.

### 4.1 Propósito científico de cada condición

| Condición | Pregunta que responde |
|---|---|
| **B0** | ¿Cuánta capacidad de localización orientada transfiere DOTA a la flota y las vistas del Perú, sin fine-tuning? Mide localización, no clasificación fina (ver `10_environment_mmrotate.md` §4.3). |
| **C1** | Línea base real de cada familia, con el ruido de etiquetas de autos estacionados y el desbalance extremo intactos. Es el denominador de $\Delta_F$. |
| **C2** | ¿El aumento clásico de YOLO logra por sí solo lo que logra limpiar las etiquetas? Es una pregunta de una sola arquitectura y por eso solo existe en F1 (justificación en `05_architecture_comparison.md` §6.2). |
| **C3** | Efecto aislado de eliminar la señal contradictoria. Es el numerador de $\Delta_F$. |

---

## 5. Hiperparámetros

### 5.1 Invariantes entre familias

| Parámetro | Valor | Motivo |
|---|---|---|
| Píxeles de entrada | Lienzo 640×640 sobre imagen fuente 640×360 | §3.1 y §3.2 |
| Dataset de entrenamiento | 43,392 frames, idéntico conjunto de nombres de archivo | Solo cambia el contenido de los píxeles entre C1 y C3 |
| Tope de épocas | **40** | Techo duro común; en la práctica no se alcanza |
| Criterio de early stopping | **`patience = 5` épocas** sin mejora, validando **cada época** | Calibrado con el piloto (§2.1). Idéntico en las tres familias, lo que hace del presupuesto de optimización una variable controlada por convergencia y no por decreto |
| `min_delta` de la mejora | 0.001 en la métrica de monitoreo | Evita que ruido de tercer decimal reinicie la paciencia y alargue la corrida sin ganancia |
| Conjunto de validación | Los 10,873 frames reales inalterados, nunca tocados por LaMa | `04_lama_cleaning.md` §6 |
| `seed` | 42 | Determinismo |
| Umbral de inferencia | `conf = 0.001` | Curva Precision-Recall completa para AP exacto |

**Por qué se valida sobre el `val` completo cada época.** La versión anterior de este documento
introducía un subconjunto de 2,000 frames para el monitoreo, por miedo a que validar costara tanto
como entrenar. El piloto desmiente ese miedo: la validación completa añade ~30 s sobre una época de
849 s (**3.5 %**) en F1, porque el `val` sí entra en RAM (7.0 GB) y la inferencia no propaga
gradientes. En F2/F3 el sobrecosto sube a ~12 %, que sigue siendo aceptable. Eliminar el
subconjunto suprime una fuente real de confusión —dos números de validación no comparables entre
sí— sin costo apreciable.

Sigue vigente la separación entre **la métrica de monitoreo** (la interna de cada framework, usada
solo para el early stopping y las curvas) y **la métrica reportada** (`Macro AP-rIoU@[0.50:0.80]`
de `src/evaluation/metric.py`, calculada una sola vez al final con el filtro de movimiento). Nunca
se reporta la primera.

### 5.2 El batch deja de ser invariante y pasa a ser diferencia declarada

El plan anterior fijaba un batch efectivo de 16 para las tres familias. Se corrige por dos motivos.
El piloto de F1 corrió con **batch 96** consumiendo 13.6 GB de los 15 GB de cada T4, es decir ya en
el techo de VRAM, y con esa configuración alcanzó `mAP50-95` ≈ 0.75; forzarlo a 16 desperdiciaría
~85 % de la VRAM disponible, multiplicaría el número de pasos de optimización y volvería a invalidar
el piloto ya ejecutado. Además, un batch de 16 no es alcanzable ni deseable de la misma forma en los
R50-FPN, cuyo consumo por imagen es muy superior.

La regla correcta es más débil pero verificable, y es la que protege el resultado principal:

> El batch se elige **por familia** para aprovechar la VRAM disponible, y es **bit a bit idéntico
> entre C1, C2 y C3 de una misma familia**. Se declara en la tabla de diferencias de
> `05_architecture_comparison.md` §5.2 junto al backbone y el optimizador.

Esto es intachable para $\Delta_F$, que es una comparación intra-familia, y es la práctica estándar
en artículos de benchmark para las comparaciones cruzadas.

### 5.3 Receta F1 — `YOLO26s-OBB` (Ultralytics)

```yaml
model: yolo26s-obb.pt        # Pesos COCO/DOTA
imgsz: 640
epochs: 40                   # Tope; el early stopping decide
patience: 5
batch: 96                    # 48 por GPU en DDP sobre 2× T4 (13.6 GB/GPU medidos)
cache: false                 # No cabe en RAM (§2.1); el piloto midió 51 img/s sin caché
workers: 4
amp: true
optimizer: AdamW
lr0: 0.001
lrf: 0.01                    # Decaimiento cosenoidal sobre el horizonte de 40 épocas
warmup_epochs: 3
weight_decay: 0.0005
seed: 42
save_period: 5               # Checkpoint cada 5 épocas para reanudar (§7)
```

Se conservan `lr0`, `optimizer` y `batch` del piloto sin tocarlos: cambiarlos obligaría a repetir
C1 y perdería la única corrida ya validada. Los únicos cambios respecto al piloto son
`epochs: 100 → 40`, `patience: 20 → 5` y `cache: true → false`, los tres justificados en §2.

### 5.4 Receta F2 y F3 — `onedl-mmrotate`

Se parte de las configs de referencia `oriented-rcnn-le90_r50_fpn_1x_dota` y
`s2anet-le90_r50_fpn_1x_dota`, con estas sobrescrituras comunes:

```python
# Geometría idéntica a la de Ultralytics (§3.2)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    dict(type='mmdet.Pad', size=(640, 640)),
    dict(type='mmdet.RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.PackDetInputs'),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)

# Cosenoidal sobre 40 épocas: análogo estructural del lrf=0.01 de Ultralytics
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=40, eta_min_ratio=0.01),
]

optim_wrapper = dict(
    type='AmpOptimWrapper',                                   # AMP, como F1
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

train_dataloader = dict(batch_size=4, num_workers=2)           # 4 por GPU × 2 GPUs = 8 efectivo

default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'))
custom_hooks = [dict(type='EarlyStoppingHook', monitor='dota/mAP',
                     patience=5, min_delta=0.001, rule='greater')]
randomness = dict(seed=42, deterministic=False)
```

Notas de diseño, cada una con su motivo:

- **Por qué cosenoidal y no el `MultiStepLR` de la receta de referencia.** Un schedule escalonado
  necesita conocer el horizonte de antemano (hitos en 8/11 para 1×, en 24/33 para 3×). Con early
  stopping el horizonte es data-driven, así que unos hitos fijos serían inalcanzados o llegarían
  demasiado tarde. El coseno degrada de forma continua y es el análogo estructural del `lrf=0.01`
  de F1, lo que además **iguala la forma del schedule entre familias** en lugar de dejarla como
  diferencia. Es una desviación deliberada de la receta de referencia y se declara como tal.
- **`lr=0.01` con batch efectivo 8** es ~2× el escalado lineal estricto desde la referencia
  (`lr=0.005` con batch 2), un valor conservador y estable en la práctica gracias al warmup de 500
  iteraciones y al `clip_grad`. **Es obligatorio validarlo con la corrida de humo de 200
  iteraciones del runbook** y, si aparece divergencia o NaN, bajar a `lr=0.005`
  **simultáneamente en C1 y C3 de esa familia**; aplicarlo a una sola condición contamina
  $\Delta_F$ de forma irreparable.
- **`batch_size=4` por GPU** y no más: los R50-FPN con RoIAlign rotado consumen mucho más que
  YOLO26s a igual resolución, y un OOM a mitad de corrida costaría horas de cuota. La corrida de
  humo confirma el pico de VRAM antes de comprometerla.
- **`checkpoint interval=1`** con `max_keep_ckpts=2`: como estas corridas pueden exceder la sesión
  de Kaggle (§6), reanudar con granularidad de una época es lo que evita perder trabajo.
- **`deterministic=False`** a propósito: el modo determinista completo en las ops de mmcv degrada
  la velocidad de forma severa. El determinismo se limita a la semilla, y se declara.

### 5.5 Aumentaciones por condición

#### F1 — Geométricas, comunes a C1, C2 y C3

Las transformaciones geométricas no crean contenido nuevo y son necesarias para que la cabeza OBB
aprenda orientaciones arbitrarias desde el dron:

```yaml
fliplr: 0.5
flipud: 0.5
degrees: 180.0
scale: 0.5
translate: 0.1
```

#### F1 — C1 y C3 (aumentación mínima)

Se desactiva todo lo que combina o pega objetos, para que el único factor que varía entre C1 y C3
sea el contenido de los píxeles del dataset:

```yaml
mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
hsv_h: 0.015
hsv_s: 0.3
hsv_v: 0.2
```

#### F1 — C2 (aumentación clásica de YOLO)

```yaml
mosaic: 1.0
mixup: 0.15
copy_paste: 0.3
erasing: 0.4
close_mosaic: 10
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

#### F2 y F3 — C1 y C3

El `train_pipeline` de §5.4, **idéntico entre C1 y C3**. Es el análogo funcional de las
aumentaciones geométricas de F1 (volteos y rotación libre), no de C2: no incluye mosaic, mixup ni
copy-paste, y por eso C2 no se replica en F2/F3.

---

## 6. Presupuesto de Cómputo

Todas las cifras se derivan del anclaje empírico de §2 (51 img/s para F1 en 2× T4). Para F2/F3 se
asume conservadoramente **1/3 del throughput de F1** para Oriented R-CNN y **~2/5** para S²A-Net,
por el costo del FPN y del remuestreo rotado de RoIs.

| Corrida | Throughput est. | Min/época (con val) | Épocas esperadas | **Horas esperadas** | Horas al tope de 40 |
|---|---|---|---|---|---|
| F1 C1 | 51 img/s | 14.7 | 11-16 | **2.7 - 3.9 h** | 9.8 h |
| F1 C2 | 51 img/s | 14.7 | 14-20 | **3.4 - 4.9 h** | 9.8 h |
| F1 C3 | 51 img/s | 14.7 | 11-16 | **2.7 - 3.9 h** | 9.8 h |
| F2 C1 / C3 | ~17 img/s | ~48 | 10-16 | **8 - 12.8 h** c/u | 32 h |
| F3 C1 / C3 | ~20 img/s | ~41 | 10-16 | **6.8 - 10.9 h** c/u | 27 h |
| B0 ×3 | Solo inferencia | — | — | **~2 h** total | — |

**Total esperado: ~45-55 h** contra una capacidad de 5 cuentas × 30 h/semana = 150 h. El margen
supera el 100 %, suficiente para absorber varias corridas fallidas.

### 6.1 Tope de seguridad para F2/F3

El escenario malo no es el OOM sino la **no convergencia**: si el `mAP` de F2/F3 mejora en el cuarto
decimal época tras época, `patience=5` nunca dispara y la corrida se va hacia las 32 h del tope.
Dos salvaguardas:

1. `min_delta=0.001` en el `EarlyStoppingHook`, que es precisamente lo que impide que una mejora
   despreciable reinicie el contador.
2. **Tope operativo de 24 épocas para F2/F3.** Si una corrida llega a la época 20 sin haber
   disparado el early stopping, se detiene manualmente en la 24 y **se aplica el mismo tope a la
   condición pareja de esa familia**. Un tope simétrico entre C1 y C3 preserva la validez de
   $\Delta_F$; un tope asimétrico la destruye. Se registra en el artículo como "24 épocas (tope)"
   en lugar de un número de early stopping.

### 6.2 Asignación en Kaggle

La restricción del plan anterior era que C1 y C3 de una familia corrieran en la **misma cuenta**
para garantizar el mismo entorno. Esa regla se relaja por una más fuerte y verificable: el entorno
de F2/F3 se instala desde el Kaggle Dataset `mmrotate-wheels` con un
`experiments/requirements-mmrotate.lock` versionado (`10_environment_mmrotate.md` §3), de modo que
es reproducible bit a bit en cualquier cuenta. **La verificación es volcar `pip freeze` en cada
corrida y hacer diff entre C1 y C3.** Con eso, repartir el par entre dos cuentas es seguro y permite
paralelizar las corridas caras.

| Cuenta | Corridas asignadas | Horas esperadas |
|---|---|---|
| **#1** | F1 C1 + F1 C3 + B0 ×3 | ~10 h |
| **#2** | F2 C1 | ~8-13 h |
| **#3** | F2 C3 | ~8-13 h |
| **#4** | F3 C1 + F1 C2 | ~11-16 h |
| **#5** | F3 C3 (+ reserva para reintentos) | ~7-11 h |

Las corridas de F2/F3 pueden exceder el límite de 9 h de una sesión interactiva. Se lanzan con
**"Save Version → Run All (background)"**, que permite hasta 12 h por ejecución, y con el protocolo
de reanudación de §7 para el caso de que ni eso baste.

### 6.3 Precedencia de ejecución

1. **F1 C1** primero, siempre: es la corrida más barata, ya tiene un piloto que la valida y
   confirma la infraestructura de datos y checkpoints antes de comprometer cuota en las corridas
   caras.
2. **Bloqueante para F2/F3:** spike de instalación de `onedl-mmrotate`
   (`10_environment_mmrotate.md` §3.1) y publicación del Kaggle Dataset de wheels. Puede avanzar en
   paralelo con el punto 1.
3. **Bloqueante para F2/F3:** conversión del dataset a formato DOTA y test de equivalencia angular
   del adaptador de predicciones. Sin esto, las corridas producirían métricas silenciosamente
   incorrectas.
4. F1 C3, y F2/F3 C1 y C3 en paralelo (el resultado principal del artículo).
5. F1 C2 y las tres evaluaciones zero-shot, que son las prescindibles si la cuota se agota.

---

## 7. Gestión de Checkpoints y Reanudación

El piloto perdió trabajo por fallos de checkpoint en Google Drive, así que este apartado deja de ser
un detalle de implementación y pasa a ser un requisito con criterios verificables.

### 7.1 Nomenclatura por corrida

Cada corrida escribe en un directorio propio y sube sus pesos con un nombre que incluye familia y
condición: `f1_c1`, `f1_c2`, `f1_c3`, `f2_c1`, `f2_c3`, `f3_c1`, `f3_c3`.

- Ultralytics: `project=/kaggle/working/runs/obb`, `name=f1_c1`.
- mmrotate: `work_dir=/kaggle/working/work_dirs/f2_c1`.
- En Drive, **una subcarpeta por corrida**. Subir todos los `last.pt` a una carpeta compartida es la
  causa más probable del fallo del piloto: los nombres colisionan y una corrida sobrescribe el
  checkpoint de otra, o descarga el ajeno al reanudar.

### 7.2 Requisitos del código de sincronización

Tres correcciones concretas sobre `src/training/`, que el runbook detalla:

1. **Guardia de rango en DDP.** El callback de subida se ejecuta en cada proceso de DDP. Sin un
   `if RANK not in (-1, 0): return`, dos procesos suben el mismo archivo a la vez y producen
   duplicados o respuestas 409 de la API de Drive.
2. **Reanudación consistente con Ultralytics.** Con `resume=True`, Ultralytics lee los argumentos
   del checkpoint e **ignora las sobrescrituras**, incluido `epochs`. Si el `last.pt` descargado
   viene de una corrida con `epochs=100`, reanudar la llevará a 100 épocas otra vez. El
   `last.pt` del piloto **no se reutiliza**: C1 se relanza desde cero con `epochs=40`, y la
   reanudación queda reservada para cortes de sesión dentro de una misma corrida ya configurada.
   Además, `save_dir` debe coincidir con el registrado en el checkpoint, lo que exige que
   `project` y `name` sean estables entre sesiones.
3. **Verificación de integridad tras la descarga.** Comparar el tamaño del archivo descargado con
   el de Drive antes de intentar cargarlo; un `last.pt` truncado hace fallar el arranque después de
   haber consumido los minutos de instalación.

### 7.3 Artefactos a preservar por corrida

`best.pt`/`best.pth`, `last.pt`/`last.pth`, `results.csv` (F1) o el `.log.json` de scalars
(F2/F3) para las curvas de pérdida del artículo, el volcado de configuración efectiva
(`args.yaml` o la config resuelta de mmengine), y el `pip freeze` exigido por §6.2.

---

## 8. Criterios de Aceptación

- [ ] Las 7 corridas completan con un early stopping registrado o con el tope documentado, sin OOM.
- [ ] C1, C2 y C3 de una misma familia comparten **exactamente** batch, semilla, optimizador,
      scheduler, pipeline de aumentación y tope de épocas. Se verifica con diff de los archivos de
      configuración volcados por cada framework.
- [ ] El `pip freeze` de C1 y C3 de una misma familia es idéntico (§6.2).
- [ ] El batch preprocesado volcado desde Ultralytics y desde mmrotate muestra el mismo vehículo con
      el mismo ancho en píxeles dentro de ±1 px (§3.2).
- [ ] El adaptador de predicciones aplica el factor de escala 3 hacia 1920×1080, verificado con una
      OBB de coordenadas conocidas (§3.1).
- [ ] Los logs (`results.csv` / `*.log.json`) están descargados para graficar curvas de pérdida.
- [ ] La métrica reportada es siempre sobre el `val` real inalterado y calculada con
      `src/evaluation/metric.py`, nunca con la métrica interna del framework.
- [ ] Los pesos descargados tienen el tamaño esperado por arquitectura (~20 MB para YOLO26s,
      ~160-330 MB para los R50-FPN), verificando integridad.
- [ ] Cada corrida registra su conteo de parámetros, GFLOPs y épocas consumidas, requeridos por las
      tablas de `07_evaluation.md`.
