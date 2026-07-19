# Plan operativo de la issue #9: métrica y filtro de movimiento

## 1. Propósito del documento

Este documento define el orden de aprendizaje, implementación, pruebas e
integración de la issue #9. Saúl y Dolly realizarán todas las etapas juntos.
No existe una división de componentes por persona: ambos deben participar en
la implementación, revisar las pruebas y poder explicar cada resultado antes
de avanzar.

Este plan es operativo. Las especificaciones científicas originales siguen
siendo las fuentes autoritativas:

1. `00_overview.md`: objetivo del estudio y dependencias generales.
2. `02_metric.md`: definición de Macro AP-rIoU, interfaz y pruebas requeridas.
3. `03_pseudo_labeling.md`: homografía, tracking y clasificación temporal.
4. `07_evaluation.md`: uso del filtro durante la evaluación final.
5. `.agents/skills/cloud-local-workflow/SKILL.md`: estructura reproducible
   para prototipos, módulos y ejecución en Colab/Kaggle.
6. El flujo ya aplicado por Álvaro en `filter_data_for_lama`: primero un
   notebook experimental, luego un módulo probado en `src/` y finalmente un
   notebook orquestador para la nube.

Antes de implementar o modificar cualquier componente de la issue #9 se debe
consultar este documento y, para los detalles matemáticos, el documento
autoritativo correspondiente.

---

## 2. Objetivo de la issue

Construir un pipeline local, modular y reproducible que:

1. elimine de las predicciones los vehículos clasificados temporalmente como
   inmóviles;
2. compare las predicciones restantes con las OBB reales mediante rIoU;
3. calcule AP usando interpolación de 101 puntos;
4. produzca Macro AP-rIoU@[0.50:0.80] sobre las nueve clases oficiales; y
5. entregue resultados detallados por clase, umbral y conteos TP, FP y FN.

El pipeline final será:

```text
Predicciones crudas (conf=0.001)
                ↓
        motion_filter.py
                ↓
Predicciones sin tracks estáticos
                ↓
             metric.py
                ↓
Macro AP-rIoU + diagnóstico detallado
```

---

## 3. Alcance y entregables

Los entregables obligatorios son:

```text
experiments/
├── notebooks/evaluation/
│   ├── prototipo_metric.ipynb
│   ├── prototipo_motion_filter.ipynb
│   └── main_colab_kaggle.ipynb
└── src/evaluation/
    ├── __init__.py
    ├── metric.py
    ├── test_metric.py
    ├── motion_filter.py
    └── test_motion_filter.py
```

`test_motion_filter.py` se añade para demostrar el criterio de aceptación del
filtro y mantener las pruebas separadas por responsabilidad.

No forman parte de esta issue:

- generación de `static_vehicles.json`;
- limpieza con LaMa;
- aumentación con IC-Light;
- entrenamiento de las condiciones experimentales;
- Grad-CAM; o
- benchmark ONNX/FP16.

`static_vehicles.json` no es una entrada de `metric.py` ni de
`motion_filter.py`. El filtro recibe detecciones crudas y matrices de
homografía.

---

## 4. Regla de trabajo conjunto

Para cada etapa se seguirá este ciclo:

1. Saúl y Dolly leen la teoría mínima de la etapa.
2. Ambos explican con un ejemplo qué entrada recibe y qué salida debe producir.
3. Ambos acuerdan la interfaz antes de escribir la implementación.
4. Experimentan juntos en el notebook de prototipo con datos pequeños,
   resultados visibles y cálculos que puedan comprobar manualmente.
5. Ambos explican el resultado experimental y deciden si la idea es correcta.
6. Trasladan la lógica validada a un módulo limpio dentro de `src/evaluation/`.
7. El notebook deja de contener la implementación definitiva y pasa a importar
   el módulo, evitando dos versiones distintas del mismo algoritmo.
8. Escriben y ejecutan las pruebas colocadas junto al módulo.
9. Ejecutan Ruff, Pyright y Pytest antes de cerrar la etapa.
10. Ambos explican por qué las pruebas demuestran que el módulo funciona.
11. Marcan la etapa como completada y recién entonces avanzan.

No se considera terminada una etapa si el código funciona pero alguno de los
dos no puede explicar su lógica.

### 4.1 Flujo obligatorio notebook → módulo → nube

Se repetirá el mismo flujo profesional utilizado por Álvaro:

```text
Notebook de prototipo (exploración y aprendizaje)
                      ↓
Validación manual de ejemplos pequeños por Saúl y Dolly
                      ↓
Módulo reutilizable en src/evaluation/
                      ↓
Pruebas colocadas junto al módulo
                      ↓
Notebook orquestador Colab/Kaggle que importa y ejecuta el módulo
                      ↓
Persistencia de resultados y checkpoints pequeños en Drive
```

Reglas:

- Los notebooks de prototipo pueden contener visualizaciones, impresiones y
  código temporal para comprender el algoritmo.
- La implementación definitiva vive exclusivamente en `src/evaluation/`.
- Los notebooks finales deben importar el paquete instalado en modo editable;
  no deben copiar las funciones de producción dentro de celdas.
- Una etapa no se migra al módulo hasta que ambos hayan entendido y validado
  manualmente el experimento correspondiente.
- El notebook orquestador debe ejecutar primero las pruebas y detener el
  pipeline si estas fallan.
- Los notebooks no sustituyen `test_metric.py` ni `test_motion_filter.py`.

### 4.2 Portabilidad entre Colab, Kaggle y ejecución local

- Los módulos no tendrán rutas fijas como `/content/drive/` o
  `/kaggle/input/`; las rutas se recibirán mediante argumentos.
- El orquestador detectará la plataforma y configurará rutas de trabajo:
  `/content/` para Colab y `/kaggle/working/` para Kaggle.
- En Kaggle, los datos de entrada se montarán desde `/kaggle/input/` o desde
  un dataset privado asociado al notebook.
- Los archivos grandes se copiarán al disco rápido de la VM antes de
  procesarlos; no se procesarán directamente dentro de Google Drive.
- La API de Google Drive se usará para guardar resultados, reportes y
  checkpoints en la cuenta central indicada por Álvaro.
- Tokens, credenciales, IDs sensibles y archivos de autenticación nunca se
  guardarán en Git. En Kaggle se proporcionarán mediante Secrets o un recurso
  privado; en Colab se leerán desde Drive.
- La ausencia de credenciales de Drive no debe impedir ejecutar las pruebas ni
  la métrica localmente; solo desactiva la sincronización remota.

---

## 5. Constantes y contrato general

Clases oficiales:

```python
CLASS_IDS = tuple(range(1, 10))
```

Umbrales oficiales:

```python
RIOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
```

Interfaz pública de la métrica:

```python
def compute_macro_ap_riou(
    predictions: dict[int, list[tuple]],
    ground_truths: dict[int, dict[str, list[tuple]]],
) -> tuple[float, dict]:
    ...
```

Formato de predicción:

```text
(frame_id, score, cx, cy, width, height, angle_deg)
```

Formato de ground truth:

```text
(cx, cy, width, height, angle_deg)
```

La métrica debe evaluar siempre las clases `1..9`. Una clase sin ground truth
recibe AP igual a `0.0`, según `02_metric.md`.

---

## 6. Bloque A: implementación de la métrica

### Etapa A0: crear el paquete y las interfaces

Tareas:

- [x] Crear `experiments/src/evaluation/__init__.py`.
- [x] Crear `metric.py` y `test_metric.py`.
- [x] Declarar constantes, alias de tipos y firmas públicas.
- [x] Confirmar que el paquete puede importarse.

Criterio de avance: el módulo se importa correctamente y ambos pueden construir
manualmente una predicción y un ground truth válidos.

### Etapa A0.1: crear el prototipo interactivo de la métrica

Archivo:

```text
experiments/notebooks/evaluation/prototipo_metric.ipynb
```

Tareas:

- [x] Crear el directorio y el notebook sin resultados pesados embebidos.
- [x] Documentar en la primera celda el objetivo, entradas y salidas.
- [x] Construir OBB sintéticas pequeñas que puedan verificarse manualmente.
- [x] Dibujar sus centros, vértices e intersecciones.
- [ ] Mostrar paso a paso los componentes de la métrica:
  - [x] rIoU.
  - [x] Matching.
  - [x] Precision y Recall.
  - [x] AP.
- [x] Registrar las conclusiones experimentales que se migrarán al módulo.

Criterio de avance: Saúl y Dolly pueden ejecutar el notebook desde el inicio,
explicar cada resultado y distinguir el código exploratorio del contrato de
producción.

### Etapa A1: validar y representar OBB

Funciones esperadas:

```text
validate_obb()
normalize_angle()
obb_to_polygon()
```

Tareas:

- [x] Experimentar primero con las transformaciones en
      `prototipo_metric.ipynb`.
- [x] Rechazar dimensiones no positivas y valores no finitos.
- [x] Normalizar ángulos mediante módulo 360.
- [x] Convertir `(cx, cy, w, h, angle)` a cuatro vértices.
- [x] Verificar que el área del polígono sea aproximadamente `w × h`.

Pruebas mínimas:

- [x] Caja válida.
- [x] Ancho cero y alto negativo.
- [x] Ángulo negativo equivalente al positivo normalizado.
- [x] Coordenadas o ángulo no finitos.

### Etapa A2: calcular rIoU

Función esperada:

```python
def rotated_iou(box_a: tuple, box_b: tuple) -> float:
    ...
```

Tareas:

- [x] Visualizar primero en `prototipo_metric.ipynb` la intersección de dos
      OBB y comprobar manualmente los casos extremos.
- [x] Calcular intersección de polígonos convexos.
- [x] Calcular unión como `área_a + área_b - intersección`.
- [x] Controlar unión cero y geometría inválida.
- [x] Garantizar un resultado dentro de `[0, 1]`.

Pruebas mínimas:

- [x] Cajas idénticas: rIoU `1.0`.
- [x] Cajas separadas: rIoU `0.0`.
- [x] Simetría `rIoU(A,B) == rIoU(B,A)`.
- [x] `-15°` y `345°` son equivalentes.
- [x] Una desviación angular reduce rIoU en una caja no cuadrada.

### Etapa A3: matching greedy y conteos

Función interna esperada:

```text
match_predictions()
```

Tareas:

- [x] Simular primero en `prototipo_metric.ipynb` el orden por score y los
      emparejamientos uno a uno.
- [x] Ordenar predicciones por score descendente de forma estable.
- [x] Comparar solo predicciones y GT de la misma clase y frame.
- [x] Elegir el GT libre con mayor rIoU.
- [x] Impedir que un GT sea utilizado dos veces.
- [x] Producir vectores TP y FP.
- [x] Calcular `FN = total_gt - TP`.
- [x] Reiniciar los matches para cada clase y umbral.

Invariantes:

```text
TP + FN = total de GT
TP + FP = total de predicciones
```

### Etapa A4: Precision, Recall y AP de 101 puntos

Función esperada:

```python
def average_precision_101(tp, fp, total_gt) -> float:
    ...
```

Tareas:

- [x] Construir primero en `prototipo_metric.ipynb` una tabla pequeña de TP,
      FP, Precision y Recall cuyo AP pueda calcularse a mano.
- [x] Calcular TP y FP acumulados.
- [x] Calcular Precision y Recall acumulados.
- [x] Evaluar niveles de recall `0.00..1.00` en pasos de `0.01`.
- [x] Usar la mejor precision disponible para cada nivel.
- [x] Usar cero cuando no existe un punto elegible.
- [x] Devolver cero si no hay ground truths.

No se debe usar integración trapezoidal: `02_metric.md` exige explícitamente
interpolación COCO de 101 puntos.

### Etapa A5: Macro AP-rIoU y diagnóstico

Tareas:

- [x] Verificar primero en `prototipo_metric.ipynb` cómo se promedian los 63
      resultados y por qué una clase sin GT aporta cero.
- [x] Evaluar `9 clases × 7 umbrales = 63` combinaciones.
- [x] Calcular AP promedio de cada clase.
- [x] Calcular Macro AP como promedio uniforme de las nueve clases.
- [x] Devolver AP por clase.
- [x] Devolver AP por clase y umbral.
- [x] Devolver TP, FP y FN por clase y umbral.
- [x] Garantizar que todos los scores estén dentro de `[0, 1]`.

Estructura mínima del detalle:

```python
{
    "ap_by_class": {},
    "ap_by_class_threshold": {},
    "counts": {},
}
```

### Etapa A6: pruebas sintéticas obligatorias

- [x] Predicción perfecta.
- [x] Predicción vacía.
- [x] Desviación angular progresiva.
- [x] Predicciones duplicadas.

Decisiones para resolver ambigüedades de `02_metric.md`:

1. Si solo las clases 1 y 2 tienen GT perfecto, el Macro AP obligatorio es
   `2/9`, porque las siete clases vacías valen cero. Se añadirá otra prueba con
   las nueve clases para obtener Macro AP `1.0`.
2. Un duplicado posterior a un TP cuenta como FP, pero AP puede seguir siendo
   `1.0` porque ya se alcanzó recall completo. La prueba verificará
   `TP=1, FP=1, FN=0`. Para demostrar una reducción de AP se añadirá un FP de
   mayor confianza antes del TP. No se alterará la métrica para forzar una
   expectativa matemáticamente incorrecta.

### Etapa A7: rendimiento

Requisito obligatorio:

```text
50,000 predicciones + 10,000 GT < 30 segundos en un hilo de CPU
```

Tareas:

- [x] Crear datos deterministas con semilla fija.
- [x] Agrupar ground truths por clase y frame.
- [x] Convertir cada OBB a polígono una sola vez.
- [x] Reutilizar rIoU entre umbrales cuando sea posible.
- [x] Evitar comparaciones entre frames diferentes.
- [x] Registrar el tiempo total del benchmark.

El Bloque A solo se considera terminado cuando todas sus pruebas y el benchmark
pasan.

---

## 7. Bloque B: filtro temporal de movimiento

### Etapa B0: contrato del filtro

Entradas:

- detecciones crudas de todos los frames de un clip; y
- matrices de homografía inter-frame.

Parámetros definidos por los planes:

```text
distancia máxima de asociación: < 30 px
duración mínima de track: >= 10 frames
desplazamiento estático: < 8 px
```

Funciones esperadas:

```text
project_centroid()
associate_detections()
build_tracks()
compute_compensated_displacement()
classify_static_tracks()
filter_static_predictions()
```

Contrato acordado:

- [x] Cada detección conserva clase, score y OBB paramétrica.
- [x] Las detecciones se agrupan por `frame_id`.
- [x] La homografía de cada frame proyecta desde el frame inmediatamente
      anterior hacia el actual; el primer frame puede no tener matriz.
- [x] Los tracks conservan referencias al índice de la detección original para
      poder filtrar duplicados sin depender de igualdad aproximada de OBB.
- [x] La salida incluye detecciones filtradas y diagnósticos auditables.
- [x] `static_vehicles.json` no forma parte de la interfaz.

### Etapa B0.1: crear el prototipo interactivo del filtro

Archivo:

```text
experiments/notebooks/evaluation/prototipo_motion_filter.ipynb
```

Tareas:

- [x] Construir una secuencia sintética corta con vehículos estáticos y
      móviles.
- [x] Dibujar centroides, IDs de track y trayectorias por frame.
- [x] Visualizar el efecto de una homografía conocida sobre los centroides.
- [x] Comparar manualmente los umbrales de 30 px, 10 frames y 8 px.
- [x] Registrar las decisiones que se trasladarán a `motion_filter.py`.

Criterio de avance: ambos pueden explicar visualmente por qué cada track se
conserva o se elimina antes de implementar el módulo definitivo.

### Etapa B1: proyección por homografía

- [x] Proyectar centroides mediante una matriz `3×3`.
- [x] Dividir las coordenadas homogéneas por `w`.
- [x] Validar matrices y resultados finitos.
- [x] Usar identidad y registrar una advertencia cuando la homografía no sea
      válida o no esté disponible.

### Etapa B2: construcción de tracks

- [x] Procesar frames en orden temporal.
- [x] Proyectar el último centroide de cada track al siguiente frame.
- [x] Asociar detecciones mediante distancia mínima greedy.
- [x] Aceptar asociaciones solo si la distancia es menor de 30 px.
- [x] Impedir que una detección pertenezca a dos tracks.
- [x] Crear tracks para detecciones sin asociación.
- [x] Conservar referencias a las detecciones originales.

### Etapa B3: clasificación y filtrado

- [x] Calcular posiciones compensadas por homografía.
- [x] Calcular la dispersión máxima dentro de cada track.
- [x] Clasificar como inmóvil si dura al menos 10 frames y su desplazamiento es
      estrictamente menor de 8 px.
- [x] Eliminar todas las detecciones de los tracks inmóviles.
- [x] Conservar los tracks móviles y los tracks de menos de 10 frames.
- [x] Devolver predicciones filtradas y diagnóstico de tracks eliminados.

### Etapa B4: pruebas del filtro

- [x] Track estático de 10 frames y menos de 8 px: eliminado.
- [x] Track de 9 frames: conservado.
- [x] Track con desplazamiento de 8 px: conservado.
- [x] Track con desplazamiento mayor de 8 px: conservado.
- [x] Cámara móvil compensada correctamente: estático eliminado.
- [x] Frames vacíos: ejecución sin errores.
- [x] Dos vehículos cercanos: sin reutilización de detecciones ni IDs.

---

## 8. Bloque C: integración final

### Etapa C1: adaptar inferencia a la interfaz

- [x] Convertir detecciones YOLO al formato público de `metric.py`.
- [x] Confirmar IDs oficiales `1..9` frente a IDs internos `0..8`.
- [x] Mantener `frame_id`, score y OBB en píxeles.
- [x] Generar predicciones crudas con `conf=0.001` fuera de `metric.py`.

### Etapa C2: filtro seguido de métrica

- [x] Ejecutar `motion_filter.py` sobre un clip sintético.
- [x] Pasar la salida a `compute_macro_ap_riou()`.
- [x] Verificar que el filtro cambia las predicciones, no los GT.
- [x] Verificar que el pipeline es determinista.
- [x] Confirmar que el mismo filtro puede aplicarse sin cambios a las seis
      condiciones experimentales.

### Etapa C3: notebook orquestador para Colab y Kaggle

Archivo:

```text
experiments/notebooks/evaluation/main_colab_kaggle.ipynb
```

El notebook seguirá la estructura del orquestador final de Álvaro, pero
separará explícitamente las operaciones dependientes de Colab y Kaggle.

Tareas:

- [x] Clonar de forma superficial y dispersa el directorio `experiments/`.
- [x] Instalar el paquete en modo editable con `.[cloud]` sin reinstalar la
      versión de PyTorch proporcionada por la plataforma.
- [x] Detectar si la ejecución ocurre en Colab, Kaggle o local.
- [x] Configurar las rutas mediante variables y argumentos, no dentro de los
      módulos.
- [x] Ejecutar `pytest src/evaluation/` antes del pipeline.
- [x] Ejecutar los módulos mediante `%run` o importarlos desde el paquete.
- [x] Ejecutar el benchmark de la métrica y guardar su reporte.
- [x] Permitir usar datasets montados en `/kaggle/input/`.
- [x] Guardar resultados intermedios primero en el disco de la VM.
- [ ] Sincronizar reportes y checkpoints pequeños con el Drive central de
      Álvaro mediante la API cuando existan credenciales.
- [ ] Probar una ejecución completa en Colab.
- [ ] Probar una ejecución completa en Kaggle.
- [x] Confirmar que una ejecución sin credenciales funciona en modo local y no
      intenta subir archivos.

No se considerará compatible con Kaggle un notebook que importe
incondicionalmente `google.colab`, dependa de `/content/drive/` o llame siempre
a `runtime.unassign()`. Esas operaciones deben quedar protegidas por la
detección de plataforma.

---

## 9. Criterio global de finalización

La issue #9 se considera completada únicamente cuando:

- [x] rIoU funciona para cajas idénticas, separadas, rotadas y ángulos negativos.
- [x] El matching es uno a uno, por clase y frame.
- [x] AP usa exactamente 101 puntos de recall.
- [x] Se evalúan las nueve clases y los siete umbrales.
- [x] Las clases sin GT reciben AP cero.
- [x] Se devuelven AP, TP, FP y FN detallados.
- [x] Pasan las pruebas sintéticas requeridas.
- [x] El benchmark cumple el límite de 30 segundos.
- [x] El filtro utiliza homografía y tracking con umbral de 30 px.
- [x] El filtro elimina únicamente tracks de al menos 10 frames y menos de 8 px.
- [x] El filtro no depende de `static_vehicles.json`.
- [x] La integración filtro → métrica funciona.
- [x] Cada algoritmo fue comprendido primero en su notebook de prototipo y
      migrado después al módulo correspondiente.
- [x] Los notebooks finales importan los módulos y no duplican su
      implementación.
- [x] El orquestador ejecuta las pruebas antes del pipeline.
- [ ] El pipeline se ejecuta tanto en Colab como en Kaggle con rutas propias de
      cada plataforma.
- [ ] Los resultados pequeños pueden persistirse en el Drive central sin
      guardar credenciales en Git.
- [ ] Saúl y Dolly pueden explicar todos los componentes.

---

## 10. Estado de progreso

Actualizar esta sección al final de cada sesión de trabajo.

| Etapa | Estado | Evidencia | Observaciones |
|---|---|---|---|
| A0: paquete e interfaces | Completada | Contrato público importable; Ruff y Pyright sin errores | La interfaz inicial ya contiene la implementación definitiva del Bloque A. |
| A0.1: prototipo de métrica | Completada | Pipeline matemático completo ejecutado localmente y en Colab sin errores; fundamentación `Ding2022DOTA` y `Yang2021KLD` documentada | Prototipo migrado al módulo; el notebook queda como explicación reproducible. |
| A1: representación OBB | Completada | `validate_obb`, `normalize_angle` y `obb_to_polygon`; 20 tests pasan | Ruff y Pyright sin errores. |
| A2: rIoU | Completada | Casos idénticos, separados, simétricos, equivalentes y angulares pasan | Intersección convexa con OpenCV y resultado acotado. |
| A3: matching greedy | Completada | Orden estable, mismo frame/clase, duplicados e invariantes probados | Los rIoU se reutilizan entre umbrales. |
| A4: AP de 101 puntos | Completada | Precision/Recall y casos AP `1.0`, `0.0` y `0.5` pasan | No se usa integración trapezoidal. |
| A5: Macro AP y detalle | Completada | 9 clases × 7 umbrales, clases vacías y diagnósticos probados | Scores inválidos y clases no oficiales se rechazan. |
| A6: pruebas sintéticas | Completada | `20 passed` en `test_metric.py` | Incluye perfecta, vacía, angular, duplicados y benchmark determinista. |
| A7: benchmark | Completada | `50,000` predicciones + `10,000` GT en un hilo: `1.427 s` en la ejecución local de referencia | Datos deterministas con semilla `2026`; el tiempo exacto se imprime en cada ejecución de pytest. |
| B0: contrato del filtro | Completada | Tipos, umbrales y funciones públicas importables; Ruff y Pyright sin errores | El contrato ya contiene la implementación definitiva validada en el prototipo. |
| B0.1: prototipo del filtro | Completada | Notebook ejecutado localmente y en Colab: estático `12 frames/0 px`, móvil `12 frames/33 px`, corto `6 frames/0 px`; asociación `29.9 px` aceptada y `30.0 px` rechazada | Las aserciones, trayectorias y proyección por homografía coinciden con el resultado esperado. |
| B1: homografía | Completada | Proyección `3×3`, división homogénea y fallback probado | Matrices ausentes, singulares o no finitas usan identidad con `RuntimeWarning`. |
| B2: tracking | Completada | Orden temporal natural, mismo `class_id`, matching global greedy `< 30 px` y referencias originales probadas | Tracks activos solo continúan desde el frame inmediatamente anterior. |
| B3: clasificación y filtro | Completada | Trayectoria compensada y dispersión máxima; umbrales `>= 10` y `< 8` probados | Medir dispersión evita que un movimiento de ida y vuelta se cancele a cero. |
| B4: pruebas del filtro | Completada | `15 passed` en `test_motion_filter.py` | Incluye límites 9/10 frames, 8 px exactos, cámara móvil, ORB, vacíos y vehículos cercanos. |
| C1: adaptación de inferencia | Completada | `Results.obb`/arrays → clase oficial, score y OBB en píxeles; inferencia `conf=0.001` probada con modelo simulado | El ángulo `xywhr` se convierte de radianes a grados. |
| C2: integración final | Completada | Filtro → métrica determinista; GT inmutables; seis condiciones y reportes JSON probados | El smoke test obtiene `1/9` y elimina 12 predicciones estáticas. |
| C3: orquestador Colab/Kaggle | Implementada; validación externa pendiente | Modo local y rutas Colab/Kaggle simuladas: `48 passed`, benchmark y smoke test sin credenciales | Faltan VMs reales, seis pesos, dataset de validación y credenciales de Drive. |

## 11. Registro de decisiones

Registrar aquí cualquier decisión que cambie una interfaz, interpretación o
criterio. No modificar silenciosamente los requisitos científicos originales.

| Fecha | Decisión | Motivo | Aprobada por |
|---|---|---|---|
| 2026-07-17 | Todo el trabajo será realizado conjuntamente por Saúl y Dolly. | Ambos deben comprender e implementar la issue completa. | Saúl y Dolly |
| 2026-07-17 | `static_vehicles.json` no será una entrada de la issue #9. | `07_evaluation.md` especifica detecciones crudas y homografías como entradas. | Según plan |
| 2026-07-17 | Las pruebas no forzarán expectativas de AP incompatibles con interpolación de 101 puntos. | Mantener corrección matemática y documentar las ambigüedades de `02_metric.md`. | Pendiente de confirmar en la issue |
| 2026-07-17 | La issue seguirá el flujo notebook de prototipo → módulo probado → notebook orquestador cloud. | Repetir el flujo de trabajo aplicado por Álvaro y permitir que Saúl y Dolly comprendan cada algoritmo antes de modularizarlo. | Indicación de Álvaro |
| 2026-07-17 | El orquestador será realmente portable entre Colab y Kaggle. | El montaje de Drive, las rutas y la desconexión del runtime son operaciones específicas de plataforma. | Indicación de Álvaro y plan cloud |
| 2026-07-19 | El prototipo del filtro queda validado para migración. | La ejecución de Colab reprodujo los conteos, tracks, desplazamientos, umbrales y visualizaciones esperados sin fallos de aserción. | Saúl y Dolly |
