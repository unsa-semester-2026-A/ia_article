# PROMPT TÉCNICO — SMART CHALLENGE 2026: IA para la Movilidad del Perú

## 1. Rol y Objetivo General

Actúa como un ingeniero de Machine Learning / Computer Vision especializado en detección de objetos orientados (OBB - Oriented Bounding Boxes). Tu tarea es ayudarme a diseñar, implementar y optimizar una solución completa para la competencia Kaggle "SMART Challenge 2026: IA para la Movilidad del Perú", organizada por el Ministerio de Transportes y Comunicaciones (MTC) del Perú.

El objetivo del reto es **detectar y clasificar vehículos en intersecciones urbanas peruanas** a partir de fotogramas de video, usando cajas delimitadoras orientadas (OBB), optimizando la métrica oficial Macro AP-rIoU.

---

## 2. Contexto de la Competencia

| Campo | Detalle |
|---|---|
| Organizador | Ministerio de Transportes y Comunicaciones (MTC) — Perú, vía DPNTRA/DGPRTM |
| Socio estratégico técnico | Artificio (startup peruana de conducción autónoma) |
| Aliados con premios | PTV Group (Alemania), Hanwha Vision (Corea del Sur), TEK Perú |
| Plataforma | Kaggle |
| Inscripciones desde | 15 de mayo de 2026 |
| Inicio de competencia | 29 de mayo de 2026 |
| Cierre de competencia | 19 de junio de 2026 |
| Definición de ganadores | 22 de junio de 2026 |
| Comunicación oficial de resultados | 23 de junio de 2026 |
| Ceremonia de premiación | 30 de junio de 2026, Lima (evento "Movilidad y Transporte en Escena: IA y Simulación 2026") |
| Elegibilidad de participantes | Estudiantes, egresados y profesionales, 18–35 años, de ingeniería civil, transportes, telecomunicaciones, software, sistemas, informática, ciencias de la computación y afines |

**Motivación institucional:** actualmente los conteos vehiculares y estudios de movimientos en intersecciones se hacen de forma manual (lento, costoso, propenso a errores). El objetivo final es generar información técnica objetiva para planificación vial, gestión del tránsito y reducción de dependencia de software importado.

---

## 3. Definición del Problema Técnico

- **Tarea:** Detección y clasificación vehicular **por fotograma**, usando oriented bounding boxes.
- **Entrada:** Clips de ~5 segundos a 10 FPS (≈50 fotogramas por clip), extraídos de intersecciones urbanas reales del Perú.
- **Evaluación:** A nivel de fotograma individual (no de video), aunque se permite usar la secuencia temporal completa del clip para mejorar predicciones (tracking, postprocesamiento temporal, modelos secuenciales).
- **Por qué OBB y no cajas horizontales:** las tomas aéreas/con perspectiva hacen que las cajas horizontales tradicionales sean insuficientes para describir la geometría real de los vehículos.
- **Uso downstream de las detecciones:** las predicciones por frame servirán como insumo base para futuros sistemas de tracking vehicular, conteo de flujos, análisis de trayectorias, matrices de giro y estudios de tráfico automatizados.

### 3.1 Desafíos propios del dataset
- Diversidad urbana: distintas zonas, infraestructura vial, densidad vehicular y configuración de intersecciones.
- Perspectiva aérea/no frontal → necesidad de bounding boxes orientados.
- Oclusión parcial, vehículos pequeños, motocicletas y mototaxis, cambios de escala.
- Desbalance de clases fuerte (algunas clases son mucho más frecuentes que otras) → la métrica oficial promedia por clase para compensarlo.

---

## 4. Clases Oficiales (9 categorías)

| ID | Clase |
|---|---|
| 1 | auto |
| 2 | combi |
| 3 | microbus |
| 4 | minibus |
| 5 | omnibus |
| 6 | articulado |
| 7 | camion |
| 8 | mototaxi |
| 9 | motocicleta |

---

## 5. Estructura de los Datos

### 5.1 Archivos entregados
- `train.zip` — imágenes de entrenamiento (.jpg)
- `test.zip` — imágenes de prueba (.jpg)
- `train.csv` — anotaciones de entrenamiento
- `sample_submission.csv` — formato de referencia para el envío

### 5.2 Convención de nombres de archivo
```
v_<video_anonimo>_<frame_index_4_digits>.jpg
```
Ejemplo:
```
v_ab12cd34ef_0000.jpg
v_ab12cd34ef_0001.jpg
v_ab12cd34ef_0002.jpg
```
- El prefijo `v_ab12cd34ef` identifica un clip de video único.
- El sufijo (`0000`, `0001`, ...) indica el índice del frame dentro del clip.
- Esto permite reconstruir la secuencia temporal de un mismo clip, incluso aunque los datos se entreguen como imágenes individuales, y usarla para mejorar las predicciones (tracking, suavizado temporal, etc.).

### 5.3 Formato de `train.csv`
Columnas: `Id, Target`

- `Id`: nombre del archivo sin extensión (ej. `v_ab12cd34ef_0000`).
- `Target`: anotaciones del frame, cada objeto en formato:
  ```
  category_id cx cy width height angle_deg
  ```
  - `category_id`: entero entre 1 y 9.
  - `cx`, `cy`: coordenadas del centro de la caja (en píxeles).
  - `width`, `height`: dimensiones de la caja (en píxeles).
  - `angle_deg`: ángulo de rotación de la caja, en grados.
  - Múltiples objetos en un mismo frame se separan con `;`.
  - Si no hay objetos, el campo debe ser exactamente `none` (no se aceptan celdas vacías).

Ejemplo de fila:
```
Id,Target
v_ab12cd34ef_0000,"1 987.86 598.84 48.84 94.88 339.94;9 1236.10 506.05 39.07 29.30 0.00"
```

---

## 6. Formato de Envío (Submission)

Para cada frame de test, se debe enviar en la columna `Target` un conjunto de detecciones con el formato:
```
score category_id cx cy width height angle_deg
```
- `score`: confianza entre 0 y 1.
- `category_id`: 1 a 9.
- `cx, cy, width, height, angle_deg`: igual que en el ground truth.
- Múltiples detecciones por frame se separan con `;`.
- Si no hay predicciones para un frame, el campo debe ser exactamente `none`.

Ejemplo:
```
0.93 1 987.86 598.84 48.84 94.88 339.94;0.81 9 1236.10 506.05 39.07 29.30 0.00
```

### 6.1 Condiciones de invalidez de una predicción
Una predicción es inválida si:
- No sigue el formato requerido.
- Contiene valores no numéricos.
- Usa `category_id` fuera del rango 1–9.
- `width <= 0` o `height <= 0`.
- Contiene coordenadas no finitas.
- Deja vacía la columna `Target` (debe decir `none`).
- Excede los límites definidos por el evaluador.

Las predicciones inválidas pueden ser descartadas o penalizadas como falsos positivos, según la implementación oficial de la métrica.

---

## 7. Métrica de Evaluación

**Métrica oficial:** `Macro AP-rIoU@[0.50:0.80]`

- Evalúa detección de objetos orientados usando **rotated IoU (rIoU)** entre cajas predichas y reales.
- Se calcula el AP para cada una de las 9 clases, en cada uno de los siguientes umbrales de rIoU:
  ```
  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
  ```
- El score final es el **promedio no ponderado** (macro) sobre las 9 clases y los 7 umbrales.
- Cada clase pesa igual → evita que el ranking sea dominado por clases frecuentes (ej. `auto`) y obliga a detectar bien también clases raras (`combi`, `microbus`, `omnibus`, `articulado`, `mototaxi`, `motocicleta`).
- Rango del score: entre 0 y 1. Mayor score es mejor.

### 7.1 Proceso de matching de predicciones
Para cada clase y cada umbral de rIoU:
1. Las predicciones se ordenan por `score` de mayor a menor.
2. Una predicción solo puede hacer match con un objeto real del **mismo frame** y **misma clase**.
3. Cada objeto real puede asignarse a lo mucho una vez.
4. Una predicción es correcta si su rotated IoU con un objeto real no asignado ≥ umbral correspondiente.
5. Predicciones duplicadas sobre el mismo objeto → falso positivo.
6. Predicciones con clase incorrecta → falso positivo.

### 7.2 Doble validación (Public vs. Private Leaderboard)
- **Public Leaderboard:** evaluado durante la competencia sobre una porción del test set; es solo referencial.
- **Private Leaderboard:** calculado al cierre sobre la porción privada del test set; **determina el resultado final**.
- El ranking final depende del Private Leaderboard, sujeto a validación de elegibilidad, cumplimiento de reglas y entrega del Dossier Técnico Digital.

---

## 8. Premios

| Puesto | Incentivos |
|---|---|
| 1° | Hasta 3 becas integrales (una por integrante) para cursos avanzados de simulación de transporte de PTV Group (hasta $900 c/u); cámara inteligente de interiores con IA de Hanwha Vision; beca integral para el Curso de Sistemas Inteligentes de Transporte en Senati (vía TEK Perú); constancia oficial del MTC y reconocimiento público |
| 2° | Hasta 3 becas integrales de PTV Group (una por integrante); beca de certificación internacional de Hanwha Vision; constancia oficial del MTC y reconocimiento público |
| 3° | Constancia oficial del MTC y reconocimiento público en canales institucionales |

---

## 9. Requisito Obligatorio: Dossier Técnico Digital (finalistas)

La confirmación de los puestos ganadores en el Private Leaderboard está sujeta a la entrega conforme de:

1. **Código fuente completo:** notebooks o scripts de entrenamiento e inferencia, limpios y comentados.
2. **Documentación técnica:** memoria descriptiva del modelo (arquitectura, hiperparámetros), pesos finales del entrenamiento y manual de instalación/entorno de ejecución.
3. **Manual de operación:** guía paso a paso para usuarios finales, orientada a procesar nuevos videos viales y configurar zonas de conteo automatizado.
4. **Análisis de resultados:** reporte en `.csv` con los flujos vehiculares calculados a partir de los datos de prueba finales.
5. **Licenciamiento GPL v3:** los equipos ganadores deben autorizar la publicación de su solución bajo licencia GNU GPL v3 (o posterior), para que sea libre, auditable y adaptable por gobiernos locales.

---

## 10. Instrucciones para la IA (tarea a resolver)

Con base en toda la especificación anterior, ayúdame a:

1. Diseñar una arquitectura de modelo adecuada para detección OBB multiclase con fuerte desbalance de clases (ej. YOLO-OBB, Oriented R-CNN, RTMDet-R, u otras alternativas), justificando la elección.
2. Proponer un pipeline de datos: carga de imágenes, parsing del formato `category_id cx cy width height angle_deg`, augmentations válidas para cajas rotadas (rotación, flip, escalado) sin romper la anotación angular.
3. Proponer una estrategia de aprovechamiento temporal del clip (aunque la evaluación es por frame) para mejorar consistencia de las predicciones, sin filtrar información de test hacia el entrenamiento de forma indebida.
4. Definir una estrategia de manejo del desbalance de clases (oversampling, focal loss, class-weighted loss, etc.), dado que la métrica es macro-promediada por clase.
5. Implementar el cálculo de rotated IoU y una función de evaluación local que replique `Macro AP-rIoU@[0.50:0.80]` para validar el modelo antes de enviar a Kaggle.
6. Generar el código de inferencia que produzca el archivo de submission en el formato exacto requerido (incluyendo el manejo del caso `none` y el separador `;`).
7. Sugerir buenas prácticas para armar el Dossier Técnico Digital exigido a los finalistas (código, documentación, manual de operación, reporte de flujos).

---

**Fuente/Cita oficial de la competencia:**
MTC SC26 en colaboración con Artificio, Darofex y Victor Flores. "SMART CHALLENGE 2026: IA para la Movilidad del Perú." https://kaggle.com/competitions/mtc-smart-challenge-ia-para-la-movilidad-del-peru, 2026. Kaggle.
