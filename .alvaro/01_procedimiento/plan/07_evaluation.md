# Evaluación y Diagnóstico (07_evaluation.md)

Este documento especifica el pipeline de evaluación cuantitativa y cualitativa para las
condiciones experimentales de las tres familias de detectores entrenadas en la Fase 06.

---

## 1. Objetivo

Evaluar el desempeño de cada modelo sobre un conjunto de validación compuesto al 100 % de datos
reales e inalterados, aplicando un filtro de movimiento temporal idéntico, y producir las tres
piezas de evidencia del artículo: la ganancia intra-familia $\Delta_F$ de la limpieza con LaMa, la
comparación de eficiencia entre familias, y el diagnóstico de saliencia que explica el mecanismo.

---

## 2. Pipeline de Inferencia y Post-Procesamiento

Los detectores entrenados identifican visualmente todos los vehículos (parados y en movimiento).
Como las anotaciones de validación solo contienen vehículos en movimiento, evaluar la inferencia
cruda reportaría una tasa artificialmente alta de falsos positivos sobre los autos estacionados.

```
┌────────────────────────────────────────────────────────────┐
│                     Inferencia Cruda                       │
│  Val real inalterado → modelo de la familia F, cond. X     │
│  → detecciones crudas con conf = 0.001                     │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│          Adaptador de Formato (src/evaluation/adapters)    │
│  Ultralytics (rad) / mmrotate (le90) → (cx,cy,w,h,deg)     │
│  → tupla canónica con angle_deg en [0, 360)                │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│               Filtro de Movimiento Temporal                │
│  1. Tracking por proximidad y solapamiento                 │
│  2. Compensación de ego-motion del dron (homografía)       │
│  3. Clasificación de vehículos inmóviles                   │
│  → descartar detecciones de vehículos estáticos            │
└──────────────────────────┬─────────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────────┐
│                     Evaluación de AP                       │
│  → Macro AP-rIoU@[0.50:0.80] (src/evaluation/metric.py)    │
└────────────────────────────────────────────────────────────┘
```

El **adaptador de formato** es un paso nuevo respecto al plan anterior y es obligatorio: sin él,
las predicciones de mmrotate llegarían al evaluador en convención le90 (radianes en
$[-\pi/2, \pi/2)$) y producirían un AP bajo pero plausible, sin lanzar ningún error. Su
especificación y su test de equivalencia están en `10_environment_mmrotate.md` §4.2.

El **filtro de movimiento** se aplica de forma estrictamente idéntica, con los mismos umbrales, a
las predicciones de todas las condiciones y de todas las familias.

---

## 3. Módulo: Filtro de Movimiento en Inferencia

- **Entrada:** detecciones crudas ya normalizadas por el adaptador, agrupadas por frame del clip
  de validación, y matrices de homografía de ego-motion del clip.
- **Proceso:**
  1. Agrupar las detecciones por frame.
  2. Enlazar predicciones entre frames con el tracking por distancia de centroide proyectado por
     homografía especificado en `03_pseudo_labeling.md`.
  3. Para cada objeto rastreado: si aparece en al menos $10$ frames y su desplazamiento neto
     compensado es $< 8$ píxeles, se clasifica como **vehículo inmóvil**.
  4. Eliminar todas las predicciones asociadas a objetos inmóviles.
  5. Retornar las predicciones filtradas.
- **Umbral de confianza:** `conf = 0.001` en la inferencia inicial, para retener las detecciones de
  baja confianza necesarias para construir la curva Precision-Recall completa.

> [!IMPORTANT]
> El filtro necesita frames consecutivos a 10 FPS, que el `val` conserva íntegros: ni el `train` ni
> el `val` se submuestrean (`06_training.md` §3).

> [!WARNING]
> No confundir la **métrica reportada** con el **valor de monitoreo** del entrenamiento. El
> monitoreo es la métrica interna de cada framework y **no** aplica el filtro de movimiento; sirve
> solo para el early stopping y las curvas de aprendizaje. Todo número que llegue al artículo se
> computa con el pipeline completo de esta sección sobre los 10,873 frames del `val`. Los dos
> valores no son comparables y no deben aparecer juntos en ninguna tabla.

> [!IMPORTANT]
> Las predicciones se generan sobre imágenes de 640×360 y el ground truth oficial está en
> 1920×1080. El adaptador debe multiplicar las coordenadas por el **factor exacto de 3** antes de
> calcular la métrica (`06_training.md` §3.1). Omitirlo produce un AP cercano a cero sin lanzar
> ningún error.

---

## 4. Tablas de Resultados a Reportar

### 4.1 Tabla principal: ganancia intra-familia (resultado central)

Es la tabla que sostiene H1 y H2 de `05_architecture_comparison.md` §4.2. Cada $\Delta_F$ compara
una familia contra sí misma, con backbone, optimizador y schedule idénticos.

| Familia | Modelo | Params (M) | GFLOPs | C1 Data cruda | C3 Data LaMa | $\Delta_F$ | $\Delta_F$ relativo |
|---|---|---|---|---|---|---|---|
| **F1** Dense end-to-end | YOLO26s-OBB | | | | | | |
| **F2** Two-stage proposal | Oriented R-CNN R50 | | | | | | |
| **F3** Single-stage aligned | S²A-Net R50 | | | | | | |

La predicción falsable de H2 es
$\Delta_{\text{F3}} > \Delta_{\text{F1}} > \Delta_{\text{F2}}$.

### 4.2 Tabla completa por condición y clase

| Familia | Condición | Macro AP-rIoU | AP@50 | AP@80 | Auto | Combi | Microbus | Minibus | Omnibus | Articulado | Camion | Mototaxi | Motocicleta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | B0 zero-shot | | | | | | | | | | | | |
| F1 | C1 cruda | | | | | | | | | | | | |
| F1 | C2 aumento clásico | | | | | | | | | | | | |
| F1 | C3 LaMa | | | | | | | | | | | | |
| F2 | B0 zero-shot | | | | | | | | | | | | |
| F2 | C1 cruda | | | | | | | | | | | | |
| F2 | C3 LaMa | | | | | | | | | | | | |
| F3 | B0 zero-shot | | | | | | | | | | | | |
| F3 | C1 cruda | | | | | | | | | | | | |
| F3 | C3 LaMa | | | | | | | | | | | | |

- **AP@50 / AP@80:** promedio macro de AP en el umbral mínimo ($0.50$) y el más estricto ($0.80$).
- **AP por clase:** promedio de AP sobre los 7 umbrales para cada una de las 9 clases.
- En las filas de B0, las columnas `Mototaxi` y `Motocicleta` serán $0$ por construcción, porque el
  vocabulario de DOTA no contiene esas categorías (`10_environment_mmrotate.md` §4.3). Debe
  aparecer anotado al pie de la tabla, no explicado como un fallo del modelo.

### 4.3 Tabla de comparación de eficiencia

Obligatoria en todo el artículo siempre que se comparen familias entre sí, para que ninguna
comparación se lea como igualdad de capacidad (`05_architecture_comparison.md` §5.2).

| Familia | Modelo | Params (M) | GFLOPs @640 | Latencia T4 FP16 (ms/frame) | FPS | Macro AP-rIoU (mejor cond.) |
|---|---|---|---|---|---|---|
| F1 | YOLO26s-OBB | | | | | |
| F2 | Oriented R-CNN R50 | | | | | |
| F3 | S²A-Net R50 | | | | | |

### 4.4 Análisis de significancia

Con una sola semilla por corrida no es posible reportar desviación estándar entre corridas, y
declararlo es preferible a insinuar una robustez que no se midió. En su lugar:

- Reportar el intervalo de confianza del AP por **bootstrap sobre los clips de validación**
  (1,000 remuestreos a nivel de clip, no de frame, para respetar la dependencia entre frames del
  mismo clip). Es una medida de la variabilidad del conjunto de evaluación, obtenible sin gastar
  cuota de GPU adicional.
- Declarar explícitamente en el artículo que $\Delta_F$ se mide con una única semilla (42) y que
  la variabilidad entre semillas no se estimó por restricción de cómputo.
- Considerar significativa una diferencia solo si excede el intervalo de confianza del 95 % del
  bootstrap.

---

## 5. Diagnóstico Cualitativo: Mapas de Saliencia

El objetivo es verificar empíricamente que la limpieza con LaMa elimina la correlación espacial con
el fondo y obliga al modelo a aprender semántica del objeto en lugar de atajos contextuales.

- **Entrada:** pesos de C1 y C3 de cada familia, y un conjunto fijo de $10$ imágenes de validación
  que contengan simultáneamente vehículos estacionados y en movimiento. Las **mismas** 10 imágenes
  para las tres familias.
- **Proceso:**
  1. Extraer gradientes y activaciones de la última capa convolucional del backbone: CSP en F1,
     `layer4` de ResNet-50 en F2 y F3.
  2. Generar los mapas Grad-CAM superpuestos.
- **Análisis esperado:**
  - **C1 (modelo ruidoso):** activaciones difusas sobre asfalto e intersecciones; el modelo
    "espera" ver o no ver vehículos según el fondo.
  - **C3 (modelo limpio):** focos estrechamente localizados sobre el chasis de los vehículos en
    movimiento, ignorando la textura limpia de la pista.
- **Valor añadido por la comparativa:** si H2 se cumple, la diferencia visual entre C1 y C3 debe
  ser **más pronunciada en F3** (Focal Loss densa) que en **F2** (negativos muestreados). Esto
  convierte la figura de Grad-CAM en evidencia del mecanismo propuesto, no en una ilustración
  decorativa, que era su rol en el plan anterior.

---

## 6. Benchmark de Latencia y Curva Precisión-Latencia

Sustenta H3 (`05_architecture_comparison.md` §4.2): si la ventaja de precisión de la familia
two-stage justifica su costo para un despliegue de monitoreo vial en tiempo real.

- **F1:** exportar a ONNX con la utilidad nativa de Ultralytics.
  ```python
  model.export(format="onnx", half=True, dynamic=True)
  ```
- **F2 y F3:** medir la latencia en PyTorch FP16 con `torch.autocast`. Se declara que F1 se mide en
  ONNX y F2/F3 en PyTorch, porque exportar detectores two-stage a ONNX con rotated RoIAlign y NMS
  poligonal introduce operadores personalizados y un riesgo de fallo que no aporta al argumento.
  Esta asimetría **favorece a F1**, y por tanto no puede sesgar la conclusión de H3 en la dirección
  que queremos demostrar; declararlo así es lo que la hace defendible.
- **Protocolo:** GPU Tesla T4 en Colab, 50 iteraciones de calentamiento y 200 de medición, batch
  sizes $1$, $4$, $8$ y $16$. Reportar media y desviación estándar en ms/frame, más los FPS.
- **Salida:** curva Precisión (Macro AP-rIoU) vs Latencia (ms/frame) con los tres modelos y los
  puntos de C1 y C3 de cada uno, para mostrar simultáneamente el costo de cada familia y el
  desplazamiento vertical que produce limpiar los datos.

---

## 7. Criterios de Aceptación

- [ ] El adaptador de formato pasa su test de equivalencia angular (rIoU ≥ 0.999) **antes** de
      evaluar cualquier modelo de F2/F3.
- [ ] El filtro de movimiento procesa las detecciones de las tres familias sin fallos de indexación
      ni desalineación de IDs, con parámetros idénticos.
- [ ] La tabla principal de $\Delta_F$ (§4.1) está completa para las tres familias.
- [ ] La tabla completa (§4.2) está llena para las 11 filas de condición×familia.
- [ ] Toda tabla que compare familias entre sí incluye parámetros, GFLOPs y latencia.
- [ ] Los intervalos de confianza por bootstrap a nivel de clip están calculados y la limitación de
      la semilla única está declarada por escrito en el artículo.
- [ ] Las figuras Grad-CAM usan las mismas 10 imágenes en las tres familias y se exportan en alta
      resolución.
- [ ] El benchmark de latencia reporta media y desviación estándar, con la asimetría
      ONNX/PyTorch declarada.
