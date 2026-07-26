# Fase 2: Aumentación Generativa de Clases Minoritarias (05_augmentation.md)

Este documento define una política reproducible para inyectar vehículos sintéticos armonizados con **IC-Light**. La síntesis complementa datos reales: no reemplaza el conjunto real ni modifica la validación.

---

## 1. Objetivo

Mitigar el desbalance extremo del dataset SMART sin convertir la distribución de entrenamiento en una colección de copias sintéticas. Cada vehículo se compone sobre un fondo de entrenamiento limpiado con LaMa, en una posición previamente ocupada por un vehículo estacionado, y se re-ilumina con IC-Light.

El resultado se usa en dos condiciones: **Mejora B** añade sintéticos a datos crudos y **Mejora C** los añade a datos LaMa. Ambas usan exactamente el mismo manifiesto de instancias sintéticas aceptadas; solo cambia el contenido de píxeles del conjunto base.

---

## 2. Construcción de candidatos

### 2.1 Recortes reales: un vehículo no equivale a 50 frames

Solo se extraen recortes desde el split **train**. Se forman tracks dentro de cada clip y se conserva un único recorte por vehículo único (track_id):

1. Se escoge el frame con mayor área visible de OBB y sin contacto con el borde; no simplemente el frame medio.
2. Se descartan OBB cuya máscara toque otra OBB anotada, estén truncadas o tengan área menor al percentil 10 de su clase.
3. Se registra track_id, clase, clip, área, relación de aspecto, ángulo y ruta del crop RGBA.
4. Cada crop se puede reutilizar como máximo diez veces con transformaciones distintas. Si una clase tiene menos de cinco tracks aprobados, se declara **insuficiencia de diversidad** y no se hacen afirmaciones fuertes sobre su mejora.

La regla evita que la frecuencia de 10 FPS se confunda con diversidad visual. En particular, un Articulado visto en cinco tracks no justifica fabricar cientos de réplicas del mismo vehículo.

### 2.2 Fondos y posiciones válidas

Los fondos también provienen solo de **train**. Un candidato debe tener una posición de vehículo estacionado eliminada por LaMa, máscara aprobada y cero solapamiento con una OBB real. El crop se coloca en esa posición $(cx, cy, \theta)$, de modo que hereda escala, orientación y soporte vial realistas.

Cada composición contiene de uno a tres objetos sintéticos, como máximo uno por clase y nunca dos veces el mismo track_id. El número de imágenes sintéticas se deriva de la asignación final; no se fija de forma arbitraria en 400.

### 2.3 Armonización y control geométrico

IC-Light (iclight_sd15_fbc.safetensors) recibe la composición y máscara a 512×512 con el prompt "outdoor urban road, daylight, traffic, aerial view"; la salida vuelve a 640×360 mediante Lanczos. La etiqueta YOLO-OBB se transforma con la misma escala, rotación y flip del crop.

Antes de aceptar una composición se verifica que no haya borde alfa visible, que la OBB quede dentro de la imagen y que no invada una instancia real. La geometría y el fondo real son esenciales: trabajos de composición controlada muestran que el alineamiento objeto-región y la coherencia visual son condiciones necesarias para que datos generativos sean útiles, no solo imágenes plausibles.

---

## 3. Política reproducible de cantidad y selección

### 3.1 Mezcla real–sintética: por qué el techo es 50%

No se entrena con imágenes 100% sintéticas. Weber, Bongartz y Roscher observaron en detección aérea de vehículos que una imagen artificial aporta menos información útil que una real y que su ganancia disminuye cuando aumenta el soporte real. Zhang, Liu y Gao hallaron que generar más sin selección puede perjudicar el detector; en su dominio el mejor punto fue 40% real, no una constante universal. Huang et al. usan muestreo equilibrado 50/50 real/sintético y concluyen con una etapa solo real.

Por tanto, se adopta un límite conservador **por clase**, no un supuesto universal para todo el dataset:

$$
\frac{S_c}{N_c + S_c} \leq 0.50
\quad\Longleftrightarrow\quad
S_c \leq N_c,
$$

donde \(N_c\) son instancias reales de *train* y \(S_c\) sintéticas aceptadas. El 50% preserva al menos la mitad de evidencia real dentro de la clase aumentada. La validación permanece 100% real y sin modificaciones.

### 3.2 Qué clases se aumentan

No se eligen simplemente “las cinco o seis últimas”. Una clase es elegible si su frecuencia en train cumple:

$$
p_c = N_c / N_{\mathrm{train}} < \tau,
\qquad \tau = 0.02.
$$

El umbral 2% identifica la cola que posee menos de 2 de cada 100 instancias y evita sintetizar para clases con soporte suficiente. Con los conteos globales actuales, las elegibles son exactamente **Combi, Microbús, Ómnibus, Articulado y Mototaxi**; quedan fuera Minibús (3.15%), Camión (5.43%), Motocicleta (7.90%) y Auto (80.03%). La decisión se recalcula con los conteos del split train antes de generar.

Mo y Yan motivan este tratamiento de desequilibrio primer plano–primer plano para vehículos orientados, pero su objetivo de balance completo no se adopta literalmente: llevar todas las clases al nivel de Auto requeriría mayoritariamente contenido sintético y destruiría la comparabilidad con el dominio real.

### 3.3 Fórmula del cupo por clase

Sea \(T=\lceil0.02N_{\mathrm{train}}\rceil\) el soporte mínimo deseado y \(U_c\) el número de tracks únicos aprobados. Para cada clase elegible:

$$
S_c^{\mathrm{sol}} =
\min\{N_c,\ \max(0,T-N_c),\ 10U_c\}.
$$

El primer término aplica el techo 50%; el segundo detiene el aumento al alcanzar 2%; el tercero limita la repetición de cada vehículo único. Es un guardarraíl experimental de diversidad, por lo que se registra y se reporta, no se atribuye a la bibliografía como una constante probada.

Como referencia de planificación, con los conteos globales disponibles (\(N=601\,934\), \(T=12\,039\)) y antes de medir \(U_c\):

| Clase | Reales de referencia | Objetivo por soporte | Máximo por 50% | Máximo antes de diversidad |
|---|---:|---:|---:|---:|
| Combi | 10,152 | 1,887 | 10,152 | **1,887** |
| Microbús | 2,802 | 9,237 | 2,802 | **2,802** |
| Ómnibus | 2,283 | 9,756 | 2,283 | **2,283** |
| Articulado | 250 | 11,789 | 250 | **250** |
| Mototaxi | 5,539 | 6,500 | 5,539 | **5,539** |
| **Total máximo** | | | | **12,761** |

La cantidad ejecutada es \(S_c=\min(S_c^{\mathrm{sol}},S_c^{\mathrm{acept}})\); puede ser menor tras el control de calidad. Con uno a tres objetos por imagen, el número de composiciones queda entre \(\lceil\sum_cS_c/3\rceil\) y \(\sum_cS_c\). Esto corrige la inconsistencia del plan anterior: 7,250 instancias no caben en “~400 imágenes” si hay solo 3–5 objetos por imagen.

### 3.4 Selección de sintéticos: generar no equivale a aceptar

Se genera un pool candidato estratificado por clase, escala, ángulo, clip y fondo. Un detector Base 1 congelado, entrenado solo con datos reales, puntúa cada objeto insertado. Se exige:

- clase predicha correcta;
- confianza ≥ 0.25;
- IoU OBB ≥ 0.50 entre predicción y etiqueta de la composición.

Las composiciones que fallen se descartan. Entre las restantes se ordenan por pérdida/compatibilidad y se retienen primero las más coherentes, conservando cuotas por estrato para no dejar solo ejemplos fáciles. El principio sigue la selección close-to-far de Zhang, Liu y Gao: un modelo de datos reales ordena sintéticos por coherencia y el corte se decide por validación, no por inspección subjetiva.

El presupuesto final se selecciona mediante un piloto reproducible de 25%, 50% y 100% de \(S_c^{\mathrm{sol}}\). Para no optimizar contra el val oficial, se reserva antes un **calibration split** fijo: 10% de los clips de train, con seed 42, sin fuga de clips hacia el train del piloto. Los crops, fondos y el score de presupuesto del piloto usan únicamente ese train/calibration interno. Solo el presupuesto con mejor Macro AP-rIoU promedio de las cinco clases elegibles pasa a Mejora B y Mejora C; recién entonces el entrenamiento final vuelve a usar todo train y se evalúa una sola vez sobre el val real oficial.

---

## 4. Estructura del dataset

~~~
synthetic_augmented/
├── images/
│   ├── v_synth_0000.jpg
│   └── ... (cantidad derivada de §3)
├── labels/
│   ├── v_synth_0000.txt
│   └── ...
└── manifest.csv
    # synthetic_id, class, source_track_id, source_clip, background_id,
    # slot_id, transform_seed, detector_score, obb_iou, accepted
~~~

Las condiciones B y C consumen el mismo manifest.csv. Así se aísla el efecto de LaMa respecto de la síntesis.

---

## 5. Criterios de aceptación

- [ ] El manifiesto registra por clase \(N_c\), \(U_c\), \(S_c^{\mathrm{sol}}\), generadas, aceptadas y porcentaje sintético final.
- [ ] Ninguna clase supera 50% de instancias sintéticas finales ni diez usos por track_id.
- [ ] Solo clases con \(p_c<2\%\) reciben síntesis; las demás quedan reales.
- [ ] Los casos con \(U_c<5\) se etiquetan como insuficientes, no se rellenan repitiendo vehículos.
- [ ] Todo sintético aceptado pasa clase, confianza e IoU OBB de §3.4.
- [ ] La selección mantiene estratos de escala, ángulo y fondo; no hay fuga desde validación.
- [ ] El presupuesto 25/50/100% y el AP por clase quedan registrados junto a la semilla, el manifiesto y el calibration split por clip; el val oficial no decide el presupuesto.
- [ ] IC-Light completa el lote sin OOM; una auditoría visual verifica máscara, sombras y ausencia de daño a vehículos reales.

---

## 6. Referencias usadas (BibTeX)

~~~bibtex
@article{Weber2021ArtificialImagesVehicleDetection,
  author  = {Weber, Immanuel and Bongartz, Jens and Roscher, Ribana},
  title   = {Artificial and Beneficial: Exploiting Artificial Images for Aerial Vehicle Detection},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2021},
  doi     = {10.1016/j.isprsjprs.2021.02.015}
}

@article{Mo2020,
  author  = {Mo, Nan and Yan, Li},
  title   = {Improved Faster RCNN Based on Feature Amplification and Oversampling Data Augmentation for Oriented Vehicle Detection in Aerial Images},
  journal = {Remote Sensing},
  year    = {2020},
  doi     = {10.3390/rs12162558}
}

@article{Zhang2026,
  author  = {Zhang, Song and Liu, Yang and Gao, Kun},
  title   = {Enhancing AVM-based Parking-Slot Detection with Synthetic Data},
  journal = {Journal of Intelligent and Connected Vehicles},
  year    = {2026},
  doi     = {10.26599/jicv.2025.9210075}
}

@article{HuangW2025,
  author  = {Huang, Weikai and Zhang, Jieyu and Jia, Taoyang and Zheng, Chenhao and Gao, Ziqi and Park, Jae Sung and Han, Winson and Krishna, Ranjay},
  title   = {Synthetic Object Compositions for Scalable and Accurate Learning in Detection, Segmentation, and Grounding},
  journal = {arXiv preprint arXiv:2510.09110},
  year    = {2025},
  doi     = {10.48550/arXiv.2510.09110}
}
~~~
