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
experiments/src/evaluation/
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
4. Implementan y revisan el código juntos.
5. Ejecutan pruebas pequeñas y comprueban manualmente al menos un resultado.
6. Ambos explican por qué las pruebas demuestran que el módulo funciona.
7. Marcan la etapa como completada y recién entonces avanzan.

No se considera terminada una etapa si el código funciona pero alguno de los
dos no puede explicar su lógica.

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

### Etapa A1: validar y representar OBB

Funciones esperadas:

```text
validate_obb()
normalize_angle()
obb_to_polygon()
```

Tareas:

- [ ] Rechazar dimensiones no positivas y valores no finitos.
- [ ] Normalizar ángulos mediante módulo 360.
- [ ] Convertir `(cx, cy, w, h, angle)` a cuatro vértices.
- [ ] Verificar que el área del polígono sea aproximadamente `w × h`.

Pruebas mínimas:

- [ ] Caja válida.
- [ ] Ancho cero y alto negativo.
- [ ] Ángulo negativo equivalente al positivo normalizado.
- [ ] Coordenadas o ángulo no finitos.

### Etapa A2: calcular rIoU

Función esperada:

```python
def rotated_iou(box_a: tuple, box_b: tuple) -> float:
    ...
```

Tareas:

- [ ] Calcular intersección de polígonos convexos.
- [ ] Calcular unión como `área_a + área_b - intersección`.
- [ ] Controlar unión cero y geometría inválida.
- [ ] Garantizar un resultado dentro de `[0, 1]`.

Pruebas mínimas:

- [ ] Cajas idénticas: rIoU `1.0`.
- [ ] Cajas separadas: rIoU `0.0`.
- [ ] Simetría `rIoU(A,B) == rIoU(B,A)`.
- [ ] `-15°` y `345°` son equivalentes.
- [ ] Una desviación angular reduce rIoU en una caja no cuadrada.

### Etapa A3: matching greedy y conteos

Función interna esperada:

```text
match_predictions()
```

Tareas:

- [ ] Ordenar predicciones por score descendente de forma estable.
- [ ] Comparar solo predicciones y GT de la misma clase y frame.
- [ ] Elegir el GT libre con mayor rIoU.
- [ ] Impedir que un GT sea utilizado dos veces.
- [ ] Producir vectores TP y FP.
- [ ] Calcular `FN = total_gt - TP`.
- [ ] Reiniciar los matches para cada clase y umbral.

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

- [ ] Calcular TP y FP acumulados.
- [ ] Calcular Precision y Recall acumulados.
- [ ] Evaluar niveles de recall `0.00..1.00` en pasos de `0.01`.
- [ ] Usar la mejor precision disponible para cada nivel.
- [ ] Usar cero cuando no existe un punto elegible.
- [ ] Devolver cero si no hay ground truths.

No se debe usar integración trapezoidal: `02_metric.md` exige explícitamente
interpolación COCO de 101 puntos.

### Etapa A5: Macro AP-rIoU y diagnóstico

Tareas:

- [ ] Evaluar `9 clases × 7 umbrales = 63` combinaciones.
- [ ] Calcular AP promedio de cada clase.
- [ ] Calcular Macro AP como promedio uniforme de las nueve clases.
- [ ] Devolver AP por clase.
- [ ] Devolver AP por clase y umbral.
- [ ] Devolver TP, FP y FN por clase y umbral.
- [ ] Garantizar que todos los scores estén dentro de `[0, 1]`.

Estructura mínima del detalle:

```python
{
    "ap_by_class": {},
    "ap_by_class_threshold": {},
    "counts": {},
}
```

### Etapa A6: pruebas sintéticas obligatorias

- [ ] Predicción perfecta.
- [ ] Predicción vacía.
- [ ] Desviación angular progresiva.
- [ ] Predicciones duplicadas.

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

- [ ] Crear datos deterministas con semilla fija.
- [ ] Agrupar ground truths por clase y frame.
- [ ] Convertir cada OBB a polígono una sola vez.
- [ ] Reutilizar rIoU entre umbrales cuando sea posible.
- [ ] Evitar comparaciones entre frames diferentes.
- [ ] Registrar el tiempo total del benchmark.

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

### Etapa B1: proyección por homografía

- [ ] Proyectar centroides mediante una matriz `3×3`.
- [ ] Dividir las coordenadas homogéneas por `w`.
- [ ] Validar matrices y resultados finitos.
- [ ] Usar identidad y registrar una advertencia cuando la homografía no sea
      válida o no esté disponible.

### Etapa B2: construcción de tracks

- [ ] Procesar frames en orden temporal.
- [ ] Proyectar el último centroide de cada track al siguiente frame.
- [ ] Asociar detecciones mediante distancia mínima greedy.
- [ ] Aceptar asociaciones solo si la distancia es menor de 30 px.
- [ ] Impedir que una detección pertenezca a dos tracks.
- [ ] Crear tracks para detecciones sin asociación.
- [ ] Conservar referencias a las detecciones originales.

### Etapa B3: clasificación y filtrado

- [ ] Calcular posiciones compensadas por homografía.
- [ ] Calcular la dispersión máxima dentro de cada track.
- [ ] Clasificar como inmóvil si dura al menos 10 frames y su desplazamiento es
      estrictamente menor de 8 px.
- [ ] Eliminar todas las detecciones de los tracks inmóviles.
- [ ] Conservar los tracks móviles y los tracks de menos de 10 frames.
- [ ] Devolver predicciones filtradas y diagnóstico de tracks eliminados.

### Etapa B4: pruebas del filtro

- [ ] Track estático de 10 frames y menos de 8 px: eliminado.
- [ ] Track de 9 frames: conservado.
- [ ] Track con desplazamiento de 8 px: conservado.
- [ ] Track con desplazamiento mayor de 8 px: conservado.
- [ ] Cámara móvil compensada correctamente: estático eliminado.
- [ ] Frames vacíos: ejecución sin errores.
- [ ] Dos vehículos cercanos: sin reutilización de detecciones ni IDs.

---

## 8. Bloque C: integración final

### Etapa C1: adaptar inferencia a la interfaz

- [ ] Convertir detecciones YOLO al formato público de `metric.py`.
- [ ] Confirmar IDs oficiales `1..9` frente a IDs internos `0..8`.
- [ ] Mantener `frame_id`, score y OBB en píxeles.
- [ ] Generar predicciones crudas con `conf=0.001` fuera de `metric.py`.

### Etapa C2: filtro seguido de métrica

- [ ] Ejecutar `motion_filter.py` sobre un clip sintético.
- [ ] Pasar la salida a `compute_macro_ap_riou()`.
- [ ] Verificar que el filtro cambia las predicciones, no los GT.
- [ ] Verificar que el pipeline es determinista.
- [ ] Confirmar que el mismo filtro puede aplicarse sin cambios a las seis
      condiciones experimentales.

---

## 9. Criterio global de finalización

La issue #9 se considera completada únicamente cuando:

- [ ] rIoU funciona para cajas idénticas, separadas, rotadas y ángulos negativos.
- [ ] El matching es uno a uno, por clase y frame.
- [ ] AP usa exactamente 101 puntos de recall.
- [ ] Se evalúan las nueve clases y los siete umbrales.
- [ ] Las clases sin GT reciben AP cero.
- [ ] Se devuelven AP, TP, FP y FN detallados.
- [ ] Pasan las pruebas sintéticas requeridas.
- [ ] El benchmark cumple el límite de 30 segundos.
- [ ] El filtro utiliza homografía y tracking con umbral de 30 px.
- [ ] El filtro elimina únicamente tracks de al menos 10 frames y menos de 8 px.
- [ ] El filtro no depende de `static_vehicles.json`.
- [ ] La integración filtro → métrica funciona.
- [ ] Saúl y Dolly pueden explicar todos los componentes.

---

## 10. Estado de progreso

Actualizar esta sección al final de cada sesión de trabajo.

| Etapa | Estado | Evidencia | Observaciones |
|---|---|---|---|
| A0: paquete e interfaces | Completada | `1 passed`; Ruff y Pyright sin errores | Contrato creado; la métrica aún no calcula rIoU. |
| A1: representación OBB | Pendiente | | |
| A2: rIoU | Pendiente | | |
| A3: matching greedy | Pendiente | | |
| A4: AP de 101 puntos | Pendiente | | |
| A5: Macro AP y detalle | Pendiente | | |
| A6: pruebas sintéticas | Pendiente | | |
| A7: benchmark | Pendiente | | |
| B0: contrato del filtro | Pendiente | | |
| B1: homografía | Pendiente | | |
| B2: tracking | Pendiente | | |
| B3: clasificación y filtro | Pendiente | | |
| B4: pruebas del filtro | Pendiente | | |
| C1: adaptación de inferencia | Pendiente | | |
| C2: integración final | Pendiente | | |

## 11. Registro de decisiones

Registrar aquí cualquier decisión que cambie una interfaz, interpretación o
criterio. No modificar silenciosamente los requisitos científicos originales.

| Fecha | Decisión | Motivo | Aprobada por |
|---|---|---|---|
| 2026-07-17 | Todo el trabajo será realizado conjuntamente por Saúl y Dolly. | Ambos deben comprender e implementar la issue completa. | Saúl y Dolly |
| 2026-07-17 | `static_vehicles.json` no será una entrada de la issue #9. | `07_evaluation.md` especifica detecciones crudas y homografías como entradas. | Según plan |
| 2026-07-17 | Las pruebas no forzarán expectativas de AP incompatibles con interpolación de 101 puntos. | Mantener corrección matemática y documentar las ambigüedades de `02_metric.md`. | Pendiente de confirmar en la issue |
