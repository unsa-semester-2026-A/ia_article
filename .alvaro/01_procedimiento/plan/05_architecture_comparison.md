# Fase 2: Comparativa entre Familias de Detectores Orientados (05_architecture_comparison.md)

Este documento reemplaza al descartado `05_augmentation.md` (IC-Light). Especifica la selección
de modelos, la taxonomía de familias arquitectónicas, las hipótesis mecanísticas y el protocolo
de comparación justa para medir si el beneficio de la limpieza de ruido de etiquetas con LaMa es
**independiente de la arquitectura del detector**.

---

## 1. Objetivo y Cambio de Alcance

El plan original medía dos intervenciones sobre los **datos** (limpieza con LaMa + aumentación
armonizada con IC-Light) usando un único detector. La aumentación generativa se retira del
alcance y se sustituye por un segundo eje experimental: el **detector**.

La pregunta de investigación pasa de *"¿cuánto ayuda limpiar y aumentar?"* a una formulación más
fuerte y más difícil de refutar:

> ¿El ruido de etiquetas por omisión sistemática (vehículos estacionados no anotados) es una
> limitación **del dataset** o una limitación **del detector**? Si limpiar las etiquetas mejora a
> tres familias arquitectónicamente disjuntas, el cuello de botella es el dataset y el resultado
> generaliza más allá de YOLO.

Esto es metodológicamente superior al alcance anterior por tres razones:

1. **Elimina el riesgo de confusión con el aumento clásico.** El resultado de IC-Light era
   difícil de separar del efecto de `copy_paste` y `mosaic`, que hacen composición de objetos por
   otros medios. La comparativa arquitectónica no tiene ese solapamiento.
2. **No introduce brecha de dominio.** IC-Light generaba píxeles sintéticos en el entrenamiento y
   se evaluaba contra un `val` real, lo que hacía que un resultado negativo fuera inconcluyente
   (¿falló la idea o falló el fotorrealismo?). Aquí todas las condiciones se entrenan sobre
   píxeles reales o reconstruidos por inpainting, nunca compuestos.
3. **Convierte el estudio en un benchmark reutilizable.** Un lector con otro detector puede
   ubicar su modelo en una de las tres familias y anticipar el beneficio esperado.

---

## 2. Corrección de la Taxonomía de Familias

La taxonomía inicialmente propuesta era:

1. Single-Stage Real-Time (Anchor-Based / End-to-End)
2. Two-Stage Proposal-Based
3. Anchor-Free / Feature-Aligned

**Esta taxonomía es incorrecta en el eje anchor y debe corregirse antes de redactar el artículo.**
El motivo es concreto y verificable:

- **YOLO26-OBB no es anchor-based.** Desde YOLOv8 la cabeza de detección de Ultralytics es
  *anchor-free*: usa asignación dinámica Task-Aligned (TAL) y regresión por Distribution Focal
  Loss sobre puntos de la rejilla, sin anchors preestablecidos. YOLO26 además es end-to-end sin
  NMS. Etiquetarlo como "anchor-based" es un error que un revisor detecta de inmediato.
- **S²A-Net, el arquetipo de "feature-aligned", sí usa un anchor.** Su Anchor Refinement Network
  parte de un anchor cuadrado por posición del feature map, hecho que el propio paper de Oriented
  RepPoints señala al compararse contra él ("*the angle-based detector presets one squared anchor
  for each feature map location*").

Es decir, el eje anchor/anchor-free **no separa** las familias 1 y 3: agruparía a YOLO26 con
Oriented RepPoints y dejaría a S²A-Net junto a los two-stage. El eje que sí produce una partición
limpia y mecanísticamente significativa es **cómo el detector accede a los features del objeto
rotado**, porque es ahí donde radica la dificultad específica de la detección orientada (los
features convolucionales están alineados a los ejes de la imagen, los objetos no).

### 2.1 Taxonomía Corregida (a usar en el artículo)

| Familia | Mecanismo de acceso a features | Asignación de etiquetas | Modelo representativo |
|---|---|---|---|
| **F1. Dense End-to-End Real-Time** | Ninguna alineación explícita: predice el OBB desde features axis-aligned de la rejilla | Densa, dinámica (TAL), sin NMS | `YOLO26s-OBB` |
| **F2. Two-Stage Proposal-Based** | Alineación **esparsa por RoI**: propuestas rotadas + rotated RoIAlign remuestrea el feature map en el marco del objeto | Esparsa y muestreada (RPN 256, RoI head 512 @ ratio 1:3) | `Oriented R-CNN` R50-FPN |
| **F3. Single-Stage Feature-Aligned** | Alineación **densa en el feature map**: AlignConv deforma el kernel según el anchor refinado + Active Rotating Filters | Densa, con Focal Loss | `S²A-Net` R50-FPN |

Las tres etiquetas descriptivas del usuario se conservan como sinónimos legibles en el texto
("una pasada", "dos pasadas", "alineada"), pero la definición formal es la de esta tabla.

---

## 3. Selección de Modelos y Justificación

### 3.1 F1 — YOLO26s-OBB (ya decidido, sin cambios)

Se mantiene la decisión de `00_overview.md` §9: `yolo26s-obb`, que aporta +2.4 mAP sobre `nano`
mientras que `medium` solo aporta +0.5 mAP al doble de costo.

### 3.2 F2 — Oriented R-CNN (ResNet-50-FPN) · ICCV 2021

**Seleccionado.** Es el arquetipo del paradigma two-stage orientado y el baseline más citado del
área.

- **Es genuinamente proposal-based orientado.** Su *oriented RPN* genera propuestas rotadas
  directamente mediante la codificación *midpoint offset* (6 parámetros de regresión por anchor),
  a costo casi nulo. No es un Faster R-CNN con un ángulo añadido al final: la propuesta que entra
  a la segunda etapa ya está rotada, y el refinamiento usa rotated RoIAlign. Esto importa para la
  comparativa: si eligiéramos un detector cuyo RPN es horizontal, la familia F2 quedaría
  representada por un mecanismo que en realidad no alinea nada en la primera etapa.
- **Rendimiento de referencia:** 75.87 % mAP en DOTA-v1.0 y 96.50 % en HRSC2016 con R-50-FPN,
  superando incluso a métodos comparables con backbone R-101-FPN; 15.1 FPS a 1024×1024 en una
  RTX 2080Ti.
- **Tiene pesos preentrenados en DOTA disponibles**, requisito indispensable para que la
  condición **Base 0 (zero-shot)** exista también en esta familia y no solo en YOLO.
- **Costo de entrenamiento asumible** en el hardware del proyecto con el schedule estándar 1×/3×.

**Alternativas evaluadas y descartadas:**

| Candidato | Motivo del descarte |
|---|---|
| **RoI Transformer** (CVPR'19) | Genera propuestas orientadas mediante un módulo aprendido de transformación RRoI, computacionalmente costoso; es precisamente el "esquema lento" que Oriented R-CNN fue diseñado para reemplazar. Aporta el mismo mecanismo conceptual con peor relación costo/precisión. |
| **ReDet** (CVPR'21) | Más preciso, pero depende de convoluciones rotation-equivariant (`e2cnn`) y de un backbone ReResNet propio. Añade una variable confundida al experimento (¿gana la familia two-stage o gana la equivarianza rotacional?) y es la instalación más frágil de todas las candidatas. |
| **Rotated Faster R-CNN** | Más barato y en versiones recientes con pérdida ProbIoU alcanza mayor mAP en DOTA que Oriented R-CNN, pero su RPN es **horizontal** y usa RoIAlign horizontal. Representa peor a la familia. Queda como **fallback** si la instalación de rotated RoIAlign falla (ver `10_environment_mmrotate.md` §5). |
| **Gliding Vertex, SCRDet** | Formulaciones de la representación del OBB más que familias arquitectónicas; no aportan un contraste de paradigma. |

### 3.3 F3 — S²A-Net (ResNet-50-FPN) · TGRS 2021

**Seleccionado.** Es el arquetipo literal del concepto "feature-aligned": *Single-shot Alignment
Network*.

- **El mecanismo es exactamente el que nombra la familia.** El Feature Alignment Module usa
  *AlignConv* para muestrear el kernel convolucional según la geometría del anchor refinado, y el
  Oriented Detection Module aplica Active Rotating Filters para producir features sensibles e
  invariantes a la orientación. El paper original aísla el aporte: AlignConv sola vale ~3 mAP
  sobre convolución estándar por solo 1.41 GFLOPs adicionales, y supera a DeformConv (71.71 %) y
  GA-DeformConv (71.33 %). Es decir, la alineación es el factor causal, no la capacidad extra.
- **Es el más beneficioso para nuestro caso específico.** El paper reporta que AlignConv mejora
  especialmente las categorías **densamente distribuidas** (`small vehicles`, `large vehicles`),
  que es literalmente la composición de nuestro dataset de tráfico urbano visto desde dron.
- **Rendimiento de referencia:** ~74.1 % mAP en DOTA-v1.0 single-scale con R-50-FPN (74.01 % en el
  texto del paper original, 74.12 % en las tablas comparativas de trabajos posteriores), a 22.6 FPS;
  es el más rápido de entrenar de los dos modelos nuevos.
- **Es el más estable de entrenar** de las candidatas anchor-free/aligned, factor decisivo bajo
  cuota limitada de GPU: un entrenamiento que diverge cuesta una corrida completa.
- **Tiene pesos DOTA disponibles** para la condición zero-shot.

**Alternativa de reserva: Oriented RepPoints (CVPR'22).** Es la candidata más fuerte que queda
fuera, y conviene documentar por qué, porque es defendible elegirla:

- Ventajas: es *estrictamente* anchor-free (representa el vehículo con un conjunto adaptativo de
  puntos y una función de conversión orientada, sin regresión de ángulo), alcanza **75.97 %** mAP
  en DOTA con R-50-FPN (**+1.85 sobre S²A-Net**), y su esquema APAA de asignación adaptativa está
  diseñado para objetos pequeños y agrupados, con resultados líderes en UCAS-AOD (90.11 %) y
  DIOR-R (66.71 %). El propio paper mide +1.39 mAP frente a la regresión angular estilo S²A-Net
  con el mismo backbone.
- Desventaja operativa: el esquema APAA introduce una evaluación de calidad de muestras durante
  el entrenamiento que lo hace más sensible al learning rate y al warmup.
- **Decisión:** S²A-Net como modelo principal de F3. Si la cuota de GPU sobra tras completar la
  matriz obligatoria, se añade Oriented RepPoints como cuarto punto (ver §6, corridas
  opcionales), lo que permitiría además desdoblar F3 en "alineación por anchor refinado" vs
  "alineación por conjunto de puntos".

---

## 4. Hipótesis del Estudio

La comparativa no es un ranking ("cuál detector es mejor"), que sería un resultado débil y
dependiente del backbone. Es una prueba de una hipótesis mecanística sobre **cómo cada familia
convierte un objeto no anotado en señal de entrenamiento**.

### 4.1 Planteamiento del mecanismo

Un vehículo estacionado sin anotar es, para el detector, una región de imagen que contiene todas
las características visuales de la clase positiva pero está etiquetada como fondo. Las tres
familias lo procesan de forma estructuralmente distinta:

| Familia | Tratamiento del vehículo no anotado | Exposición esperada al ruido |
|---|---|---|
| **F3. S²A-Net** | Negativo denso ponderado por **Focal Loss**, que por diseño *amplifica* el peso de los negativos difíciles. Un objeto real etiquetado como fondo es el negativo más difícil que existe: el gradiente que más recibe es justamente el equivocado. | **Máxima** |
| **F1. YOLO26s-OBB** | Negativo denso: toda posición de la rejilla no asignada por TAL contribuye a la pérdida de clasificación, sin sobre-ponderación focal explícita de los casos difíciles. | **Intermedia** |
| **F2. Oriented R-CNN** | Negativo **muestreado**: el RPN entrena con ~256 anchors muestreados y la cabeza RoI con ~512 propuestas a ratio positivo 1:3. La mayoría de las regiones de fondo, incluidos los autos estacionados, simplemente no entra al batch en una iteración dada, diluyendo estadísticamente el ruido. | **Mínima** |

### 4.2 Hipótesis formales

- **H1 (Universalidad).** La limpieza con LaMa incrementa el Macro AP-rIoU@[0.50:0.80] en las
  **tres** familias. Consecuencia: el ruido de etiquetas por omisión es una limitación del
  dataset, no un artefacto de la arquitectura elegida.
- **H2 (Orden de magnitud del beneficio).** La ganancia $\Delta_F = \text{AP}(\text{C3}) -
  \text{AP}(\text{C1})$ ordena las familias según su exposición al ruido:
  $$\Delta_{\text{S²A-Net}} > \Delta_{\text{YOLO26}} > \Delta_{\text{Oriented R-CNN}}$$
  Esta es la predicción falsable del estudio y su aporte original: vincula un tipo de ruido de
  anotación con un mecanismo de asignación de etiquetas concreto.
- **H3 (Compensación precisión/latencia).** Bajo la métrica operativa del MTC y con el filtro de
  movimiento aplicado, la ventaja de precisión de la familia two-stage no compensa su costo de
  latencia para un despliegue de monitoreo vial en tiempo real. Se sustenta con la curva
  Precisión vs Latencia de `07_evaluation.md` §6.

### 4.3 Interpretación de un resultado negativo

El diseño está construido para que **cualquier** desenlace sea publicable, condición que el plan
anterior no cumplía:

- Si se cumple H1 y H2: resultado principal completo (universalidad + explicación mecanística).
- Si se cumple H1 pero no H2: el beneficio de limpiar es universal pero no está gobernado por el
  esquema de muestreo de negativos; obliga a discutir qué otro factor lo modula (capacidad del
  backbone, schedule) y sigue validando la contribución central.
- Si H1 falla en alguna familia: hallazgo aún más interesante, porque identifica una arquitectura
  intrínsecamente robusta al ruido de anotación por omisión, lo que es un resultado accionable
  para cualquiera que trabaje con datasets parcialmente anotados.

---

## 5. Protocolo de Comparación Justa

Una comparación entre frameworks distintos es el punto más atacable del artículo. Estas reglas
son de cumplimiento obligatorio.

### 5.1 Invariantes (idénticos en todas las corridas)

| Invariante | Valor | Motivo |
|---|---|---|
| Split train/val | Nivel de clip, `seed=42`, 80/20 (870/218 clips) según `01_data_preparation.md` §2.2 | Evita fuga de datos entre frames del mismo fondo |
| Conjunto de validación | Imágenes **reales inalteradas**, nunca tocadas por LaMa | Regla metodológica de `04_lama_cleaning.md` §6 |
| Píxeles de entrada | Lienzo 640×640 con letterbox sobre imagen fuente de 640×360 | Comparabilidad y factibilidad (`06_training.md` §3.1); verificado volcando un batch preprocesado de cada framework |
| Dataset de entrenamiento | Los 43,392 frames completos, mismo conjunto de nombres de archivo | Solo el contenido de los píxeles cambia entre C1 y C3 |
| Criterio de early stopping | `patience = 5` épocas, `min_delta = 0.001`, validando cada época, tope de 40 | Hace del presupuesto de optimización una variable controlada por convergencia y no por decreto |
| Métrica reportada | `Macro AP-rIoU@[0.50:0.80]` de `src/evaluation/metric.py` | **Nunca** el mAP interno de Ultralytics ni el de mmrotate |
| Filtro de movimiento | Idéntico, mismos umbrales, aplicado post-inferencia | `07_evaluation.md` §3 |
| Umbral de confianza | `conf = 0.001` | Curva Precision-Recall completa para AP exacto |

### 5.2 Diferencias declaradas (no se ocultan, se reportan)

Igualar todo entre frameworks es imposible y forzarlo produciría modelos mal entrenados. La
práctica estándar en artículos de benchmark es entrenar cada detector con su receta de referencia
y **declarar** las diferencias. Se reportarán explícitamente en el artículo:

| Aspecto | YOLO26s-OBB | Oriented R-CNN / S²A-Net |
|---|---|---|
| Backbone | CSP propio, ~11 M parámetros | ResNet-50-FPN, ~36-41 M parámetros |
| Optimizador | AdamW, `lr0=0.001` | SGD momentum 0.9, `lr=0.01` |
| Batch | 96 (48 por GPU en 2× T4, 13.6 GB/GPU medidos) | 8 (4 por GPU) |
| Preentrenamiento | COCO + DOTA | ImageNet + DOTA |
| Épocas consumidas | Las que decida `patience=5` (tope 40) | Ídem, con tope operativo de 24 (`06_training.md` §6.1) |

**El batch no es un invariante, y esto es deliberado.** Se elige por familia para aprovechar la VRAM
disponible y es idéntico entre las condiciones de una misma familia, que es lo que $\Delta_F$
necesita. La justificación completa, incluida la medición de VRAM que la sustenta, está en
`06_training.md` §5.2.

La asimetría de parámetros es la objeción más previsible. Se neutraliza así:

1. Se añade una columna obligatoria de **parámetros y GFLOPs** a toda tabla de resultados, para
   que ninguna comparación se lea como "igualdad de capacidad".
2. La conclusión central **no depende de comparar familias entre sí**, sino de comparar cada
   familia **contra sí misma** (C1 vs C3). Cada $\Delta_F$ es una comparación intra-arquitectura
   con backbone, optimizador y schedule idénticos, así que es inmune a esta objeción. Esto es
   deliberado y debe explicitarse en la sección de metodología del artículo.
3. Si la cuota de GPU lo permite, se añade una corrida de control con `YOLO26m-obb` (~2× los
   parámetros de `s`) para acotar cuánto del gap absoluto es capacidad y cuánto es paradigma.

### 5.3 Presupuesto de entrenamiento común

En lugar de fijar un número igual de épocas para las tres familias, se fija un **criterio de parada
igual**: `patience = 5` épocas sin mejora superior a `min_delta = 0.001`, validando cada época,
con un tope duro de 40 épocas.

Esto es más defendible que la simetría de épocas del plan anterior. Un número igual de épocas
favorece arbitrariamente a la arquitectura que converge más lento, mientras que un criterio de
parada igual entrena cada detector **hasta que deja de aprender**, que es la comparación que
interesa. La calibración de `patience=5` no es arbitraria: sale de la corrida piloto de F1
documentada en `06_training.md` §2, donde 33 épocas más allá de la época 6 aportaron ~0.01 de mAP.

Consecuencias que deben aparecer en el artículo:

- El schedule de learning rate es **cosenoidal en las tres familias** (`lrf=0.01` en Ultralytics,
  `CosineAnnealingLR` en mmrotate), en lugar del `MultiStepLR` de la receta de referencia de
  mmrotate. Un schedule escalonado exige conocer el horizonte de antemano, lo que es incompatible
  con una parada data-driven. La desviación se declara.
- Se reporta el número real de épocas consumidas por cada corrida, y si una alcanzó un tope se
  reporta como tope y no como convergencia.

---

## 6. Matriz Experimental

### 6.1 Volumen de datos: dataset completo, sin submuestrear

La versión anterior de este documento proponía un submuestreo temporal 1:5 (de 43,400 a ~8,700
frames) para hacer viable el cronograma. **Ese submuestreo queda descartado**, y la razón es que la
medición del piloto lo volvió innecesario y contraproducente.

El error del razonamiento anterior fue estimar el costo como
$\text{frames} \times \text{épocas fijas}$. Con early stopping el costo real es el número de
**muestras procesadas hasta converger**, que es una propiedad del problema y no del tamaño de la
época. El piloto de F1 convergió tras ~260 k muestras (época 6 sobre 43,392 frames). Submuestrear 5×
haría cada época 5× más corta, pero se necesitarían ~5× más épocas para procesar esas mismas 260 k
muestras: el tiempo total es equivalente. Lo único que cambia es que `patience`, medida en épocas,
quedaría descalibrada.

Es decir, el submuestreo aportaba un riesgo metodológico real —una corrida de control adicional, una
objeción previsible sobre el recorte de datos, un módulo de software más que testear— a cambio de un
ahorro de cómputo nulo. La derivación numérica completa está en `06_training.md` §2.1.

Se conserva el artefacto versionado único `smart-640` (y su variante `smart-640-lama`) con los
43,392 frames de entrenamiento y 10,873 de validación, para que las siete corridas consuman
exactamente los mismos bytes.

### 6.2 Matriz de condiciones

| Condición | Dataset de entrenamiento | YOLO26s-OBB (F1) | Oriented R-CNN (F2) | S²A-Net (F3) |
|---|---|---|---|---|
| **Base 0** Zero-shot | Ninguno (pesos DOTA + mapeo de clases) | Eval | Eval | Eval |
| **C1** Data cruda | `smart-640` original | Entrena | Entrena | Entrena |
| **C2** Aumento clásico | `smart-640` original | Entrena | — | — |
| **C3** Data limpia LaMa | `smart-640-lama` | Entrena | Entrena | Entrena |

**Total: 7 entrenamientos y 3 evaluaciones zero-shot.** La corrida E1 de control de escala del plan
anterior desaparece junto con el submuestreo que existía para cuantificar.

**Por qué C2 existe solo en F1.** El aumento "clásico" del plan original (`mosaic=1.0`,
`mixup=0.15`, `copy_paste=0.3`, `erasing=0.4`) es específico del pipeline de Ultralytics; las
recetas de referencia de mmrotate para detección aérea usan `RandomRotate` + `RResize` +
`RandomFlip` y no incluyen mosaic ni mixup. Replicarlo en F2/F3 significaría implementar esas
transformaciones a mano, lo que introduce una variable de implementación propia justo en la
condición donde se necesita comparabilidad. C2 responde a una pregunta de una sola arquitectura
—*¿el aumento clásico logra por sí solo lo que logra limpiar las etiquetas?*— y se reporta como
tal.

### 6.3 Corridas opcionales (solo si sobra cuota, en este orden)

1. `Oriented RepPoints` C1 y C3 → desdobla F3 en alineación por anchor vs por conjunto de puntos.
2. `YOLO26m-obb` C1 → control de capacidad para el gap absoluto entre familias (§5.2).
3. C2 para F2 y F3 con las transformaciones geométricas de mmrotate, declarando que no es
   equivalente a C2 de F1.

### 6.4 Presupuesto de cómputo

Anclado en el throughput medido del piloto (51 img/s para F1 en 2× Tesla T4), no en extrapolaciones:

| Corrida | Modelo | Épocas esperadas | Horas esperadas |
|---|---|---|---|
| C1, C3 | YOLO26s-OBB | 11-16 | ~2.7-3.9 h c/u |
| C2 | YOLO26s-OBB | 14-20 | ~3.4-4.9 h |
| C1, C3 | Oriented R-CNN R50 | 10-16 | ~8-12.8 h c/u |
| C1, C3 | S²A-Net R50 | 10-16 | ~6.8-10.9 h c/u |
| Zero-shot ×3 | Solo inferencia | — | ~2 h total |
| **Total** | | | **~45-55 h** |

Contra una capacidad de 5 cuentas × 30 h/semana = 150 h/semana el margen supera el 100 %, muy
holgado para absorber corridas fallidas. Los R50-FPN cuestan aproximadamente 2.5-3× por imagen lo
que cuesta YOLO26s a la misma resolución, por el costo del FPN y del remuestreo rotado de RoIs. El
desglose por época, el tope de seguridad para F2/F3 y la asignación por cuenta están en
`06_training.md` §6.

---

## 7. Entregables de Software Nuevos

La comparativa introduce requisitos de código que no existían en el plan anterior:

1. **Conversor de formato `YOLO-OBB → DOTA`** (`src/data_preparation/dota_converter.py`): mmrotate
   consume anotaciones en formato DOTA (`x1 y1 x2 y2 x3 y3 x4 y4 nombre_clase dificultad`, en
   píxeles absolutos), mientras el pipeline actual produce YOLO-OBB normalizado. La conversión es
   una desnormalización más un mapeo de índice a nombre de clase, y debe tener test de ida y
   vuelta (`round-trip`) con tolerancia de 1e-6.
2. **Adaptadores de predicciones** (`src/evaluation/adapters.py`): normalizan la salida de
   Ultralytics y de mmrotate al formato único que consume `compute_macro_ap_riou`, es decir
   `(frame_id, score, cx, cy, w, h, angle_deg)` con ángulo en grados en $[0, 360)$. Es el punto
   de mayor riesgo de error silencioso del proyecto: mmrotate usa por defecto la convención
   **le90** (ángulo en radianes en $[-\pi/2, \pi/2)$) y Ultralytics otra. Requiere test de
   equivalencia que verifique que un mismo OBB codificado en las dos convenciones produce
   rIoU = 1.0 tras la conversión. El adaptador es además el punto donde se aplica el **factor de
   escala 3** de 640×360 a las coordenadas oficiales de 1920×1080 (`06_training.md` §3.1).
3. **Runner de entrenamiento para mmrotate** (`src/training/trainers/train_mmrotate.py`):
   análogo a `train_base_1.py`, con la misma sincronización a Drive y checkpoint cada época.
4. **Mapeo de clases DOTA → SMART** para las evaluaciones zero-shot: `small-vehicle → auto`,
   `large-vehicle → {camion, omnibus, microbus, minibus, combi}`. El mapeo es intrínsecamente
   ambiguo y debe documentarse como limitación de Base 0, no como un defecto de la medición.

---

## 8. Criterios de Aceptación

- [ ] Las 7 corridas completan sin OOM, con un early stopping registrado o con el tope documentado.
- [ ] El test de equivalencia de convención angular (le90 ↔ grados $[0,360)$) pasa con
      rIoU ≥ 0.999 para un conjunto de 100 OBB aleatorios, incluidos casos límite de ángulo
      negativo y de $w < h$.
- [ ] El conversor a DOTA supera el test de ida y vuelta con error máximo < 1e-6 en píxeles.
- [ ] El Macro AP-rIoU de las tres familias se computa con **el mismo módulo** y sobre el **mismo
      `val` real inalterado**; queda prohibido reportar métricas internas de los frameworks.
- [ ] Toda tabla de comparación entre familias incluye columnas de parámetros, GFLOPs y latencia
      (ms/frame en T4 FP16).
- [ ] Ningún `clip_id` aparece en los dos splits a la vez.
- [ ] El batch preprocesado volcado desde Ultralytics y desde mmrotate muestra el mismo vehículo con
      el mismo ancho en píxeles dentro de ±1 px (`06_training.md` §3.2).
- [ ] El `pip freeze` de C1 y C3 de una misma familia es idéntico (`06_training.md` §6.2).
