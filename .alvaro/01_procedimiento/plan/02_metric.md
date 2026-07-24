# Implementación de la Métrica Macro AP-rIoU (02_metric.md)

Este documento contiene la especificación de diseño, lógica matemática, interfaces de software y pruebas unitarias sintéticas para la métrica oficial de evaluación: **Macro AP-rIoU@[0.50:0.80]**.

---

## 1. Objetivo
Garantizar una evaluación local exacta e idéntica a la utilizada en la plataforma de Kaggle para el SMART Challenge 2026. La métrica evalúa la precisión media macro-promediada sobre 9 clases vehiculares y 7 umbrales de Rotated Intersection over Union (rIoU).

---

## 2. Especificación Matemática

### 2.1 Rotated Intersection over Union (rIoU)
Dadas dos cajas delimitadoras orientadas representadas de forma paramétrica:
- Caja de Predicción: $B_p = (cx_p, cy_p, w_p, h_p, \theta_p)$
- Caja de Ground Truth (GT): $B_g = (cx_g, cy_g, w_g, h_g, \theta_g)$

El rotated IoU se define como:

$$\text{rIoU}(B_p, B_g) = \frac{\text{Área}(P(B_p) \cap P(B_g))}{\text{Área}(P(B_p) \cup P(B_g))}$$

Donde $P(B)$ es el polígono cerrado de 4 vértices asociado a la caja parametrizada. 

* **Cálculo de la Intersección:** Se obtiene mediante la intersección geométrica de polígonos convexos en 2D. Se sugiere el uso de la biblioteca **Shapely** en Python (`shapely.geometry.Polygon`) o el método de OpenCV `cv2.rotatedRectangleIntersection`.
* **Cálculo de la Unión:** Se computa como la suma de las áreas de ambos polígonos menos el área de su intersección:

$$\text{Área}(P(B_p) \cup P(B_g)) = \text{Área}(P(B_p)) + \text{Área}(P(B_g)) - \text{Área}(P(B_p) \cap P(B_g))$$

---

### 2.2 Proceso de Matching y Clasificación de Detecciones
Para cada clase $c \in [1, 9]$ y cada umbral de rIoU $t \in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]$:

1. **Ordenamiento:** Las predicciones correspondientes a la clase $c$ se ordenan por su score de confianza de mayor a menor.
2. **Matching Frame-a-Frame:** Una predicción solo puede asignarse a un objeto real de la misma clase dentro del mismo frame.
3. **Asignación Greedy:**
   - Para cada predicción ordenada, buscar el objeto real de la misma clase que maximice el rIoU.
   - Si el rIoU máximo $\ge t$ y el objeto real no ha sido asignado a otra predicción previa con mayor score:
     - Marcar la predicción como **Verdadero Positivo (TP)**.
     - Registrar el objeto real como "asignado" (no puede usarse de nuevo).
   - De lo contrario (rIoU $< t$ o el objeto real ya está asignado), marcar como **Falso Positivo (FP)**.
4. **Falsos Negativos (FN):** Los objetos reales que quedan sin asignar al finalizar el proceso se registran como Falsos Negativos.

---

### 2.3 Cálculo de Average Precision (AP) por Clase y Umbral
Una vez clasificados los TP y FP para un par (clase $c$, umbral $t$):

1. Generar vectores acumulativos de TP y FP.
2. Calcular la curva de Precision-Recall a nivel acumulativo:
   - $\text{Recall}(i) = \text{TP}_{acum}(i) / N_{gt}$
   - $\text{Precision}(i) = \text{TP}_{acum}(i) / (\text{TP}_{acum}(i) + \text{FP}_{acum}(i))$
3. Calcular el AP mediante la **interpolación de 101 puntos** (estándar de COCO):

$$\text{AP}_{c, t} = \frac{1}{101} \sum_{r \in \{0.0, 0.01, ..., 1.0\}} \max_{\tilde{r} \ge r} \text{Precision}(\tilde{r})$$

Si la clase $c$ no tiene instancias de Ground Truth en el split de validación, su $\text{AP}_{c,t}$ se define como $0.0$ para evitar indeterminaciones.

---

### 2.4 Puntuación Macro AP-rIoU@[0.50:0.80]
El score final se obtiene promediando uniformemente sobre las 9 clases vehiculares y los 7 umbrales evaluados:

$$\text{Score Final} = \frac{1}{9} \sum_{c=1}^{9} \left( \frac{1}{7} \sum_{t \in T} \text{AP}_{c, t} \right)$$

Donde $T = \{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80\}$.

---

### 2.5 Conjunto final de metricas para reportar

Para el articulo y los reportes generados en Kaggle se usara el siguiente
conjunto compacto de metricas. Las metricas de deteccion se calculan despues
de aplicar el filtro de movimiento definido en `07_evaluation.md`.

| Grupo | Metrica | Descripcion | Uso |
|---|---|---|---|
| Principal | `Macro mAP-rIoU@[0.50:0.80]` | Promedio uniforme de AP sobre 9 clases y 7 umbrales rIoU. | Score principal del articulo. |
| Principal | `mAP-rIoU@0.50` | Promedio macro de AP con rIoU minimo de 0.50. | Deteccion con criterio flexible. |
| Principal | `mAP-rIoU@0.80` | Promedio macro de AP con rIoU estricto de 0.80. | Calidad de localizacion OBB. |
| Principal | `AP por clase` | AP promedio de cada clase sobre los 7 umbrales. | Diagnosticar clases minoritarias y desbalance. |
| Analisis | `Precision` | `TP / (TP + FP)` por umbral. | Medir falsos positivos. |
| Analisis | `Recall` | `TP / (TP + FN)` por umbral. | Medir objetos no detectados. |
| Analisis | `F1-score` | Media armonica de precision y recall. | Balance precision-recall. |
| Analisis | `TP, FP, FN` | Conteos agregados por umbral. | Auditoria de errores. |
| Filtro | `% predicciones eliminadas` | Fraccion de detecciones removidas por el filtro. | Auditar cuanto limpia el filtro. |
| Filtro | `% tracks estaticos` | Fraccion de tracks clasificados como inmoviles. | Auditar comportamiento temporal. |
| Eficiencia | `FPS` | Frames por segundo teoricos de inferencia. | Costo computacional. |
| Eficiencia | `VRAM pico` | Maxima memoria GPU usada durante inferencia. | Requerimiento de hardware. |

El nombre `macro_score` queda como compatibilidad interna del codigo, pero en
tablas y texto cientifico debe reportarse como
`Macro mAP-rIoU@[0.50:0.80]` o `macro_map_riou_50_80`.

---

## 3. Especificación de la Interfaz del Módulo

### 3.1 Función de Entrada
```python
def compute_macro_ap_riou(
    predictions: dict[int, list[tuple]], 
    ground_truths: dict[int, dict[str, list[tuple]]]
) -> tuple[float, dict]:
```

#### Parámetros:
* `predictions`: Un diccionario indexado por `class_id` ($1$ a $9$) que contiene una lista de tuplas con el formato:
  `[(frame_id: str, score: float, cx: float, cy: float, w: float, h: float, angle_deg: float), ...]`
* `ground_truths`: Un diccionario indexado por `class_id` ($1$ a $9$) que contiene otro diccionario indexado por `frame_id`, que a su vez contiene una lista de cajas reales:
  `{frame_id: [(cx: float, cy: float, w: float, h: float, angle_deg: float), ...]}`

#### Retorno:
1. `macro_score`: Un valor punto flotante en el rango $[0.0, 1.0]$.
2. `detailed_results`: Un diccionario con métricas detalladas para diagnóstico:
   - AP por clase (promedio de umbrales)
   - AP para cada par (clase, umbral)
   - Conteos acumulados de TP, FP y FN por clase y umbral.

---

## 4. Plan de Pruebas Unitarias (Tests Sintéticos de Validación)

El módulo debe ser validado con un conjunto de datos sintéticos controlados. Se especifican los siguientes 4 casos de prueba:

### Test 1: Predicción Perfecta
* **Configuración:** 10 objetos de Ground Truth distribuidos en 5 imágenes. Mezcla de clase 1 y clase 2. Predicciones idénticas a los Ground Truths con scores de confianza de 1.0.
* **Resultado Esperado:** $\text{AP}_{c, t} = 1.0$ para todas las combinaciones. Score Macro Final = $1.0$ (o $0.222$ si se consideran las clases vacías como $0.0$; la interfaz debe manejar adecuadamente si se calcula sobre las clases existentes o sobre las 9 clases obligatoriamente. Para el challenge, las 9 clases siempre se promedian, por lo que si una clase no tiene predicciones ni GT, su AP es 0).

### Test 2: Predicción Vacía
* **Configuración:** Ground Truth idéntico al Test 1. Diccionario de predicciones vacío.
* **Resultado Esperado:** $\text{AP}_{c, t} = 0.0$ para todas las clases. Score Macro Final = $0.0$.

### Test 3: Desviación Angular Progresiva
* **Configuración:** Ground Truth con OBB alineado a $0^\circ$. Predicciones con el mismo centro y tamaño, pero con desvíos angulares rotados en $+10^\circ$, $+20^\circ$, $+30^\circ$, y $+45^\circ$.
* **Resultado Esperado:** El rIoU debe decrecer progresivamente. A $+45^\circ$, el rIoU caerá por debajo del umbral de $0.50$ (para una caja no cuadrada), por lo que el AP debe ser $0.0$. La prueba debe verificar que el score final decrece de forma monótona conforme aumenta la rotación.

### Test 4: Penalización de Predicciones Duplicadas
* **Configuración:** 1 objeto real en el frame. 2 predicciones idénticas con scores de confianza de $0.95$ y $0.80$ sobre la misma ubicación.
* **Resultado Esperado:** La predicción con score de $0.95$ se marca como TP. La predicción duplicada con score $0.80$ se marca como FP. El AP resultante debe ser menor a $1.0$, demostrando que la asignación es no-repetitiva.

---

## 5. Criterios de Aceptación y Rendimiento
- [ ] Todos los tests sintéticos de validación (1 al 4) deben pasar sin errores de aserción.
- [ ] El cálculo de rIoU debe ser robusto ante ángulos negativos (ej. $-15^\circ$ debe ser equivalente a $345^\circ$).
- [ ] La función debe poder computar los resultados sobre un set de $50,000$ predicciones y $10,000$ GTs en un tiempo inferior a $30$ segundos en un solo hilo de CPU.
