# Entorno e Interoperabilidad para las Familias F2 y F3 (10_environment_mmrotate.md)

Este documento especifica el entorno de ejecución, el procedimiento de instalación reproducible y
los adaptadores de formato necesarios para entrenar Oriented R-CNN (F2) y S²A-Net (F3) descritos
en `05_architecture_comparison.md`. Es el documento de mayor riesgo técnico del plan y su
validación bloquea todo el eje de comparación arquitectónica.

---

## 1. Objetivo

Disponer de un entorno único, reproducible e instalable en menos de 5 minutos dentro de un
notebook de Kaggle o Colab, capaz de entrenar los dos modelos nuevos con pesos preentrenados de
DOTA, y de exportar sus predicciones al formato que consume el evaluador propio del proyecto.

---

## 2. Decisión de Framework

**Seleccionado: `onedl-mmrotate`** (fork mantenido por VBTI del MMRotate de OpenMMLab).

| Criterio | Justificación |
|---|---|
| **Un solo toolbox para F2 y F3** | Oriented R-CNN, S²A-Net y Oriented RepPoints están los tres en el mismo repositorio con configs de referencia. Esto es indispensable para la comparación justa: comparten pipeline de datos, optimizador, scheduler y código de evaluación, así que la diferencia medida entre F2 y F3 es arquitectónica y no de implementación. |
| **PyTorch 2.0+** | La rama principal soporta PyTorch 2.x, a diferencia del MMRotate original de OpenMMLab, cuya última versión (0.3.4, febrero 2023) exige `mmcv-full>=1.5.3,<1.8.0` y `mmdet<3.0.0`, incompatibles con el PyTorch que traen preinstalado los entornos de Kaggle y Colab. |
| **Mantenimiento activo** | El MMRotate original está descontinuado. El fork corrige bugs y compatibilidad con versiones nuevas de PyTorch. |
| **Model zoo con pesos DOTA** | Habilita la condición Base 0 (zero-shot) para F2 y F3, que de otro modo solo existiría en YOLO. |
| **Imagen Docker oficial** | Existe `vbti/onedl-mmrotate-cu129-torch2100` en Docker Hub, útil para reproducir el entorno en la VM local con GTX 1070 sin pelear con dependencias. |

### 2.1 Alternativas evaluadas

| Alternativa | Veredicto |
|---|---|
| **`oriented-det`** (PyTorch puro, sin MMCV) | Muy atractivo en instalación: es PyTorch puro, sin kernels CUDA propios ni dependencia de MMCV, con pesos DOTA en Hugging Face y CLI `odet`. **Descartado como framework principal porque solo trae Oriented R-CNN, Rotated Faster R-CNN y Rotated RetinaNet: ningún modelo feature-aligned.** No cubre F3, así que no evita la dependencia de mmrotate y además nos obligaría a comparar F2 y F3 entre dos frameworks distintos. Queda como **fallback de F2** (§5, nivel 3). |
| **MMRotate 0.3.4 original** con `torch==1.13` + `mmcv-full==1.7.2` | Funciona y tiene wheels precompiladas oficiales, pero exige degradar PyTorch dentro del notebook, lo que arrastra a `torchvision`, y en Kaggle rompe la GPU preinstalada con frecuencia. Queda como **fallback de entorno** (§5, nivel 2). |
| **Detectron2** (RRPN / rotated boxes) | Soporta cajas rotadas, pero su soporte OBB es secundario (nació para detección de texto), no tiene modelos feature-aligned ni pesos DOTA, y también requiere compilación. Sin ventajas sobre mmrotate. |

---

## 3. Procedimiento de Instalación

### 3.1 Spike de validación (bloqueante, primer día del eje arquitectónico)

**No se debe planificar ninguna corrida antes de completar este spike.** Toda la matriz de F2 y F3
depende de que `mmcv.ops` importe correctamente en Kaggle: Oriented R-CNN, S²A-Net y Oriented
RepPoints están marcados en el README de `onedl-mmrotate` como modelos que **requieren
`onedl-mmcv` con soporte de ops**, y además todos los detectores necesitan `batched_nms` de
`mmcv.ops` incluso en inferencia.

Criterios de éxito del spike, en un notebook de Kaggle con GPU activa:

```python
import torch, mmcv, mmcv.ops, mmdet, mmrotate   # ninguna excepción
print(torch.__version__, torch.cuda.is_available())
from mmrotate.apis import init_detector, inference_detector  # smoke test
```

Y una corrida de 50 iteraciones de `oriented_rcnn_r50_fpn_1x_dota_le90` sobre 20 imágenes del
dataset convertido, verificando que la pérdida decrece y que no hay OOM a `imgsz=640`, `batch=4`.

**Registrar en el issue correspondiente:** versión exacta de `torch`, `mmcv`, `mmengine`, `mmdet`,
`mmrotate`, versión de CUDA, y si `mmcv.ops` vino de wheel o se compiló.

### 3.2 Kaggle Dataset de wheels precompiladas (obligatorio)

Compilar `mmcv` con ops desde código fuente tarda entre 40 y 90 minutos en una máquina de Kaggle,
tiempo que se consume de la cuota de GPU en **cada** sesión y que se pierde cada vez que el
notebook se reinicia. Con 7 corridas más reintentos, esto solo puede costar decenas de horas de
cuota.

Procedimiento:

1. En **una sola** sesión (puede ser Colab, para no gastar cuota de Kaggle), construir las wheels:
   ```sh
   pip wheel --no-deps -w /content/wheels onedl-mmengine onedl-mmcv onedl-mmdetection onedl-mmrotate
   ```
2. Subir el directorio como Kaggle Dataset privado **`mmrotate-wheels`**, compartido con las 5
   cuentas del equipo.
3. En cada notebook de entrenamiento, instalar sin red y sin compilar:
   ```sh
   pip install --no-index --find-links=/kaggle/input/mmrotate-wheels onedl-mmrotate
   ```
4. Congelar el resultado de `pip freeze` en el repositorio como `experiments/requirements-mmrotate.lock`
   para que las 7 corridas usen versiones idénticas. Una diferencia de versión entre la corrida C1
   y la C3 de la misma familia invalidaría el $\Delta_F$, que es el resultado principal del
   artículo.

### 3.3 Gestión de dependencias en el paquete `experiments`

Se añade un extra opcional en `experiments/pyproject.toml`, separado de `cloud` y `local`, porque
las dependencias de mmrotate no deben instalarse en los notebooks de YOLO (riesgo de conflicto de
versión de `numpy`, ya bloqueado en `numpy==1.26.4` por el crash de ABI documentado en el commit
`0138f81`):

```toml
[project.optional-dependencies]
mmrotate = [
    "onedl-mmengine",
    "onedl-mmcv",
    "onedl-mmdetection",
    "onedl-mmrotate",
]
```

Las versiones exactas se fijan **después** del spike, con los valores que se comprueben
funcionales, no antes.

---

## 4. Interoperabilidad de Formatos

Este es el punto donde un error pasa inadvertido y contamina los resultados: una conversión
angular equivocada no lanza excepción, solo baja el AP de forma plausible.

### 4.1 Anotaciones: `YOLO-OBB` → `DOTA`

El pipeline actual produce YOLO-OBB normalizado (`class_idx x1 y1 x2 y2 x3 y3 x4 y4` con
coordenadas en $[0,1]$, clases 0-indexadas, según `01_data_preparation.md` §2.3). mmrotate consume
el formato DOTA:

```
x1 y1 x2 y2 x3 y3 x4 y4 nombre_clase dificultad
```

con coordenadas en **píxeles absolutos**, **nombre** de clase en texto (no índice) y un campo de
dificultad que se fija en `0` para todas las instancias.

La conversión es una desnormalización por $(W, H)$ más un mapeo índice→nombre usando exactamente
los nombres de `smart_dataset.yaml`. Requisitos:

- Los frames vacíos (`Target == "none"`) generan un archivo `.txt` **vacío**, igual que en YOLO.
- Test de ida y vuelta obligatorio: `YOLO → DOTA → YOLO` con error máximo < 1e-6.

### 4.2 Predicciones: convención angular (riesgo alto)

Las tres fuentes usan convenciones distintas para el ángulo del OBB:

| Fuente | Convención | Rango |
|---|---|---|
| `train.csv` del SMART Challenge / evaluador propio | Grados, antihorario respecto al eje X | $[0, 360)$ |
| Ultralytics YOLO-OBB | Radianes | $[0, \pi/2)$ |
| mmrotate (configs `*_le90`) | Radianes, **le90** | $[-\pi/2, \pi/2)$ |

Un OBB es invariante ante un giro de $180°$ y ante el intercambio simultáneo de $w \leftrightarrow h$
con un giro de $90°$, de modo que la conversión no es una simple multiplicación por $180/\pi$: hay
que normalizar la representación antes de comparar. El módulo `src/evaluation/adapters.py` debe:

1. Convertir toda predicción a la tupla canónica
   `(frame_id, score, cx, cy, w, h, angle_deg)` con `angle_deg` en $[0, 360)$, que es lo que
   `compute_macro_ap_riou` espera según `02_metric.md` §3.1.
2. Pasar un **test de equivalencia**: para 100 OBB aleatorios, codificarlos en le90 y en la
   convención del challenge, convertirlos con el adaptador y verificar `rIoU ≥ 0.999`. El conjunto
   de prueba debe incluir explícitamente ángulos negativos, ángulos cercanos a $\pm 90°$ y cajas
   con $w < h$, que son los casos donde las convenciones se separan.
3. Reutilizar el `rIoU` ya implementado en `src/evaluation/metric.py`; el test no debe
   reimplementar geometría, para que un error en el rIoU no se cancele con un error en el
   adaptador.

Este test es también una salvaguarda de `02_metric.md` §5, que ya exige robustez ante ángulos
negativos ("$-15°$ debe ser equivalente a $345°$").

### 4.3 Mapeo de clases para las evaluaciones zero-shot

DOTA tiene 15 categorías, de las cuales solo dos son vehículos terrestres. El mapeo hacia las 9
clases del SMART Challenge es necesariamente sobreyectivo:

| Clase DOTA | Clases SMART |
|---|---|
| `small-vehicle` | `auto` |
| `large-vehicle` | `combi`, `microbus`, `minibus`, `omnibus`, `articulado`, `camion` |
| — | `mototaxi`, `motocicleta` (sin equivalente en DOTA) |

Consecuencias que deben declararse en el artículo como limitación del diseño de Base 0, no como
defecto de la medición:

- `mototaxi` y `motocicleta` tendrán AP = 0 por construcción en las tres evaluaciones zero-shot,
  porque el vocabulario de DOTA no las contiene.
- Toda predicción `large-vehicle` es ambigua entre 6 clases. Se adopta la regla determinista de
  asignarla a la clase mayoritaria de ese grupo en el dataset (`camion`, con 32,668 instancias
  según `01_data_preparation.md` §2.1) y se documenta la regla.
- El Base 0 mide entonces **capacidad de localización orientada transferida**, no capacidad de
  clasificación fina. Esa es la lectura correcta y la única defendible.

---

## 5. Escalera de Contingencia

Se aplica en orden; cada nivel solo se activa si el anterior falla en el spike.

**Nivel 1 (plan principal).** `onedl-mmrotate` + `onedl-mmcv` con ops, instalado desde el Kaggle
Dataset de wheels. Cubre F2 (Oriented R-CNN) y F3 (S²A-Net) en un mismo framework.

**Nivel 2 (entorno legacy).** MMRotate 0.3.4 original con `torch==1.13.1`, `mmcv-full==1.7.2`
(wheel oficial precompilada para cu117) y `mmdet==2.28.2`. Mismos dos modelos y misma comparación
justa; solo cambia el entorno. Costo: degradar PyTorch dentro del notebook y revalidar que la GPU
sigue disponible.

**Nivel 3 (framework mixto, comparabilidad reducida).** F2 con `oriented-det` (PyTorch puro, sin
MMCV, con checkpoint `oriented_rcnn_dota_le90_1x` preentrenado) y F3 con mmrotate. Se debe
declarar en el artículo que F2 y F3 corren en frameworks distintos, lo que debilita la comparación
**entre** familias pero **no** afecta el resultado principal, que es intra-familia
($\Delta_F$ = C3 − C1 con el mismo framework en ambas condiciones). Esta es la razón por la que el
diseño de `05_architecture_comparison.md` §5.2 apoya la conclusión en comparaciones
intra-arquitectura: sobrevive a este escenario.

**Nivel 4 (reducción de alcance).** Sustituir F2 por `Rotated Faster R-CNN` (disponible en ambos
frameworks, sin necesidad de rotated RoIAlign) declarando que su RPN es horizontal y que por tanto
representa la familia two-stage de forma parcial. Es la última opción porque debilita
precisamente el mecanismo que define a F2.

---

## 6. Criterios de Aceptación

- [ ] `import mmcv.ops` y `import mmrotate` funcionan en Kaggle con GPU, con versiones registradas.
- [ ] La instalación desde el Kaggle Dataset `mmrotate-wheels` completa en < 5 minutos sin acceso
      a red y sin compilar.
- [ ] `experiments/requirements-mmrotate.lock` está versionado y las 7 corridas lo usan sin
      divergencias.
- [ ] El conversor `YOLO-OBB → DOTA` pasa el test de ida y vuelta con error < 1e-6.
- [ ] El adaptador de predicciones pasa el test de equivalencia angular con rIoU ≥ 0.999,
      incluyendo casos de ángulo negativo, ángulo cercano a $\pm 90°$ y $w < h$.
- [ ] Una corrida de humo de 50 iteraciones de Oriented R-CNN y de S²A-Net decrece la pérdida sin
      OOM a `imgsz=640`.
- [ ] Los pesos DOTA de ambos modelos están descargados y respaldados en Google Drive, para que
      una caída del model zoo no bloquee el cronograma.
