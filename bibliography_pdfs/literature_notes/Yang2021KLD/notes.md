# Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence

- **Key**: Yang2021KLD
- **Year**: 2021 (arXiv v1: junio 2021, publicado NeurIPS 2021)
- **Venue**: NeurIPS 2021 (Advances in Neural Information Processing Systems)

## Resumen

Este artículo aborda el problema de la pérdida de regresión para la detección de objetos rotados de alta precisión. Los detectores rotados existentes heredan el paradigma de detección horizontal añadiendo simplemente el parámetro de ángulo θ a la pérdida Smooth L1, lo que optimiza cada parámetro de forma independiente y resulta en detección imprecisa, especialmente para objetos con grandes ratios de aspecto. El artículo propone un cambio de paradigma: en lugar de partir de la detección horizontal (caso especial) hacia la rotada (caso general), plantea diseñar la pérdida de regresión directamente para el caso general rotado, de modo que la detección horizontal sea su caso degenerado. La solución consiste en convertir la caja rotada B(x, y, w, h, θ) en una distribución Gaussiana 2D N(μ, Σ) y usar la Kullback-Leibler Divergence (KLD) entre la distribución predicha y la ground truth como pérdida de regresión. El análisis del gradiente demuestra que KLD ajusta dinámicamente la importancia de cada parámetro según las características del objeto (aspect ratio, escala, ángulo), implementando un mecanismo de auto-modulación. Además, se demuestra que KLD es scale-invariant y se degenera en la pérdida ln-norm para el caso horizontal. Los experimentos en siete datasets (DOTA-v1.0/v1.5/v2.0, UCAS-AOD, HRSC2016, ICDAR2015, MLT, MSRA-TD500) con múltiples detectores confirman la superioridad de KLD, alcanzando 80.63% mAP en DOTA-v1.0.

---

## Secciones y Subsecciones

### 1. Introduction

La introducción motiva el diseño desde primeros principios de una pérdida de regresión para detección rotada. Se observa que la mayoría de detectores rotados heredan el paradigma inductivo de la detección horizontal: adaptan la pérdida Smooth L1 añadiendo un parámetro de ángulo extra. Este enfoque optimiza los cinco parámetros (x, y, w, h, θ) de forma independiente, lo que resulta problemático pues el under-fitting de cualquier parámetro afecta severamente la precisión final. El ejemplo concreto es el objeto con gran aspect ratio: un pequeño error en θ causa una caída drástica en IoU, pero la pérdida Smooth L1 no pondera el ángulo más pesadamente en esos casos. El artículo propone un enfoque deductivo: diseñar una pérdida para el caso general rotado que se degenegre coherentemente al caso horizontal especial.

* **Problemas atacados**: La optimización independiente de parámetros en la pérdida de regresión de detectores rotados impide la detección de alta precisión, especialmente en objetos con grandes ratios de aspecto.
* **Limitaciones de ese entonces**: Smooth L1 y GWD (Gaussian Wasserstein Distance) no acoplan completamente todos los parámetros; GWD no es scale-invariant y optimiza el centro independientemente.
* **Soluciones alcanzadas**: KLD como pérdida de regresión implementa acoplamiento de cadena entre todos los parámetros, auto-modulación adaptativa de gradientes, scale-invariance, y coherencia con el caso horizontal como degeneración.

---

### 2. Background

Esta sección revisa los trabajos relacionados y caracteriza los dos paradigmas de diseño de pérdidas de regresión para detección rotada.

#### 2.1. Related Works

Se revisa la detección horizontal (Faster R-CNN, RetinaNet, FCOS, etc.) que usa cajas horizontales con pérdidas ln-norm o IoU-based. Se presenta la detección rotada como extensión de estos detectores con cajas orientadas (OBB), y se clasifica la literatura según dos metodologías: inductiva (de horizontal especial a rotado general) y deductiva (de rotado general a horizontal especial).

* **Problemas atacados**: Ubicar la propuesta en el contexto de la literatura existente, distinguiendo entre los dos paradigmas de diseño.
* **Limitaciones de ese entonces**: La mayoría de detectores rotados seguían el paradigma inductivo, sin plantear si la pérdida de regresión es coherente con el caso horizontal.
* **Soluciones alcanzadas**: La clasificación inductivo/deductivo proporciona un marco conceptual claro para comparar métodos y justificar el enfoque del artículo.

#### 2.2. Inductive Thinking of Loss Design: from Special Horizon to General Rotation Detection

Se describe formalmente la pérdida de regresión inductiva estándar: se predicen los offsets (tx, ty, tw, th) y se añade el offset angular tθ = f(θp - θa), optimizando todos con Smooth L1 de forma independiente. La crítica central es que los cinco parámetros tienen diferentes escalas y unidades, y su importancia varía según el tipo de objeto, pero la pérdida no lo refleja. Además, el under-fitting de cualquier parámetro degrada directamente la métrica IoU final.

* **Problemas atacados**: Formalizar la limitación fundamental de la pérdida ln-norm independiente para objetos rotados.
* **Limitaciones de ese entonces**: La extensión directa del caso horizontal al rotado mediante Smooth L1 ignora las interacciones entre parámetros y su importancia diferencial según el tipo de objeto.
* **Soluciones alcanzadas**: La formalización matemática permite comparar cuantitativamente con KLD y motivar el nuevo diseño.

#### 2.3. Deductive Thinking of Loss Design: from General Rotation to Special Horizon Detection

Se introduce el enfoque deductivo: convertir la caja rotada B(x, y, w, h, θ) en una distribución Gaussiana 2D N(μ, Σ) mediante la transformación Σ = RΛR⊤. Se revisa la Gaussian Wasserstein Distance (GWD) como trabajo previo que sigue este paradigma: GWD descompone en distancia del centro (independiente) y términos de acoplamiento de h, w, θ, siendo una pérdida "semi-acoplada". Se identifica que GWD no acopla completamente el centro con el resto y no es scale-invariant, lo que limita su rendimiento en detección de alta precisión.

* **Problemas atacados**: Explorar el paradigma deductivo como alternativa a la pérdida inductiva, y identificar las limitaciones de la propuesta previa GWD.
* **Limitaciones de ese entonces**: GWD no es completamente acoplada (el centro se optimiza independientemente) y carece de scale-invariance, propiedad crucial para detección.
* **Soluciones alcanzadas**: El análisis de GWD motiva el uso de KLD, que sí posee acoplamiento completo y scale-invariance.

---

### 3. Proposed Approach

Esta sección es el núcleo técnico del artículo y presenta la pérdida KLD para detección rotada de alta precisión.

**Kullback-Leibler Divergence.** Se calcula la KLD entre dos distribuciones Gaussianas 2D Np(μp, Σp) y Nt(μt, Σt) como pérdida de regresión. La KLD se descompone analíticamente en términos que revelan el acoplamiento de parámetros: Dkl(Nt||Np) tiene acoplamiento de cadena entre todos los parámetros (x, y acoplados con θ a través de la matriz de covarianza; h, w acoplados con ∆θ; θ afectado por el aspect ratio hp/wp). En contraste, Dkl(Np||Nt) y GWD son semi-acopladas.

**Análisis de alta precisión.** Se deriva el gradiente de KLD respecto a cada parámetro para analizar el mecanismo de auto-modulación. Para el centro (x, y): los pesos 1/wt² y 1/ht² hacen que el modelo ponga más atención al desplazamiento en la dirección más estrecha del objeto. Para h, w: la penalización es mayor cuando la dimensión predicha difiere más de la target, y está acoplada con ∆θ. Para θ: el gradiente es mayor cuando el aspect ratio del objeto es mayor (hp²-wp²), lo cual es la propiedad clave para detección de alta precisión en objetos elongados. Este mecanismo de auto-modulación es fundamentalmente diferente a Smooth L1 (gradientes independientes) y GWD (solo semi-acoplado).

**Scale invariance.** Se demuestra formalmente que para cualquier transformación afín M de rango completo, Dkl(Np'||Nt') = Dkl(Np||Nt). En particular, con M = kI, se prueba la scale-invariance. Esta propiedad es crucial para detección, donde los objetos pueden aparecer a múltiples escalas, y la pérdida no debe ser sensible a ella. Ni Smooth L1 ni GWD poseen esta propiedad.

**Caso especial horizontal.** Se demuestra algebraicamente que cuando θ = 0°, la KLD se degenera en una combinación de normas L1 y L2 sobre los desplazamientos normalizados (∆tx, ∆ty, ∆tw, ∆th), que es coherente con la pérdida estándar de detección horizontal (excepto por el factor de normalización wt vs. wa).

**Variantes de KLD.** Se introducen variantes simétricas: Dkl_min, Dkl_max, Jensen-Shannon divergence (Djs) y Jeffrey's divergence (Djef). Los experimentos muestran que la asimetría de KLD tiene impacto mínimo en el rendimiento.

**Pérdida de regresión final.** La pérdida de regresión se define como Lreg = 1 - 1/(τ + f(D)), con f(D) = log(D+1) y τ = 1 como configuración óptima. Esta normalización suaviza el crecimiento de KLD y hace la pérdida más expresiva. La pérdida multi-tarea combina Lreg y focal loss para clasificación.

* **Problemas atacados**: Diseñar una pérdida de regresión que acopla todos los parámetros, es auto-modulada, scale-invariant, y coherente con el caso horizontal.
* **Limitaciones de ese entonces**: No existía una pérdida de regresión rotada que reuniera simultáneamente acoplamiento completo, scale-invariance y degeneración coherente al caso horizontal.
* **Soluciones alcanzadas**: KLD logra un mecanismo de auto-modulación que ajusta adaptativamente la importancia del gradiente de θ según el aspect ratio, de h, w según ∆θ, y del centro según la escala, todo en una pérdida unificada y matemáticamente fundamentada.

---

### 4. Experiment

#### 4.1. Datasets and Implementation Details

Los experimentos se realizan en siete datasets: DOTA-v1.0/v1.5/v2.0 (detección aérea, 15-18 categorías), UCAS-AOD (2 categorías, 1510 imágenes), HRSC2016 (barcos), ICDAR2015, MLT y MSRA-TD500 (texto en escenas). La implementación usa TensorFlow con ResNet50, SGD sobre 8 GPUs, 8 imágenes por minibatch. Se entrena por 20 épocas con lr inicial 5e-4 (reducida ×10 en épocas 12 y 16). Se usaron dos detectores base: RetinaNet y R3Det.

* **Problemas atacados**: Validar la generalidad y consistencia de KLD en múltiples datasets y detectores.
* **Limitaciones de ese entonces**: Las pérdidas de regresión previas se evaluaban en uno o dos datasets; la validación en siete datasets con dos detectores proporciona evidencia más robusta.
* **Soluciones alcanzadas**: El protocolo experimental extensivo demuestra que KLD supera consistentemente a Smooth L1 y GWD en todos los datasets y detectores evaluados.

#### 4.2. Ablation Study and Further Comparison

Se realizan ablaciones sobre la forma de la pérdida (Dkl crudo, f(Dkl), Lreg con τ), variantes de KLD (Dkl_min, Dkl_max, Djs, Djef), y normalización. Resultados clave: (a) la normalización con f(D) = log(D+1) y τ = 1 es óptima (85.25% en HRSC2016); (b) las variantes simétricas no mejoran significativamente, confirmando que la asimetría de KLD no es un problema; (c) la normalización Eq. 18 es genuinamente útil y no es la responsable de la mejora (demostrado por el drop al normalizar Smooth L1).

El experimento de detección de alta precisión es central: en HRSC2016 (barcos con gran aspect ratio), KLD mejora AP75 en +23.97% sobre Smooth L1 con RetinaNet y +33.96% con R3Det. En métricas de alta precisión (AP75, AP85, AP50:95), KLD supera consistentemente a GWD y Smooth L1 en todos los datasets, con las mayores ventajas en escenarios de alta precisión y objetos de gran aspect ratio.

* **Problemas atacados**: Demostrar cuantitativamente el mecanismo de auto-modulación de KLD y su ventaja específica en alta precisión.
* **Limitaciones de ese entonces**: Las métricas estándar (AP50) no evidencian completamente las ventajas de precisión; el análisis con AP75 y AP50:95 es necesario para revelar el comportamiento en alta precisión.
* **Soluciones alcanzadas**: Las ablaciones confirman que la ventaja de KLD proviene del mecanismo de acoplamiento, no de la normalización, y que las mejoras en alta precisión son consistentes y robustas.

#### 4.3. Comparisons with the State-of-the-Art Methods

La combinación RetinaNet-KLD-R50 (single-scale) alcanza 75.28% mAP en DOTA-v1.0, superando modelos multi-scale previos. R3Det-KLD-R50 logra 77.36%. Con backbone más grande y multi-scale testing, KLD alcanza 80.63% mAP. En los datasets de texto escena (ICDAR2015, MSRA-TD500, MLT), KLD muestra mejoras consistentes de 3-10% sobre Smooth L1 y 2-6% sobre GWD. En detección horizontal (MS COCO), KLD es competitivo o ligeramente superior a Smooth L1 y GIoU, con mejora de +0.6% en RetinaNet-AP.

* **Problemas atacados**: Validar que KLD no solo mejora en ablaciones sino que establece nuevo estado del arte en benchmarks competitivos.
* **Limitaciones de ese entonces**: Los métodos previos (GWD, CSL, DCL, etc.) no alcanzaban el rendimiento de detección de alta precisión logrado por KLD.
* **Soluciones alcanzadas**: KLD establece nuevo estado del arte en DOTA-v1.0 (80.63% con multi-scale) y mejora todos los métodos base evaluados.

---

### Discussions

Se identifican las limitaciones del método: KLD no puede aplicarse directamente a detección de cuadriláteros arbitrarios (no paramétricos como rotated boxes). También se señalan posibles impactos negativos: la mejora en detección precisa de objetos orientados puede facilitar aplicaciones en sensado remoto, aviación o drones con fines cuestionables. Se concluye que KLD provee una pérdida de regresión unificada, con base matemática sólida, auto-modulada y scale-invariant para detección rotada, coherente con la detección horizontal como caso especial.

* **Problemas atacados**: Contextualizar las limitaciones y posibles impactos del trabajo para una comunicación científica responsable.
* **Limitaciones de ese entonces**: La representación en caja rotada paramétrica (x, y, w, h, θ) limita la aplicabilidad directa de KLD a representaciones alternativas como cuadriláteros.
* **Soluciones alcanzadas**: El artículo establece un marco teórico y práctico sólido para pérdidas de regresión en detección rotada, con la posibilidad de extender KLD a otras representaciones como trabajo futuro.

---

### A. Appendix

#### A.1. Proof of Scale Invariance of KLD

Se demuestra formalmente que para cualquier transformación lineal M de rango completo, Dkl(MNp||MNt) = Dkl(Np||Nt) usando propiedades algebraicas de matrices (tr, det, inversa). La demostración cubre el caso general de invarianza afín, del cual la scale-invariance es un caso particular con M = kI.

* **Problemas atacados**: Proporcionar la prueba matemática rigurosa de scale-invariance de KLD, que la diferencia de GWD y Smooth L1.
* **Limitaciones de ese entonces**: No existía una prueba formal de scale-invariance para pérdidas de regresión de detección rotada.
* **Soluciones alcanzadas**: La prueba formal establece que KLD es afín-invariante, propiedad que garantiza comportamiento consistente independientemente de la escala de los objetos.

#### A.2. Analysis of Dkl(Nt||Np)'s High-Precision Detection

Se extiende el análisis de gradientes a la forma simétrica Dkl(Nt||Np) y se demuestra que posee propiedades similares de auto-modulación, aunque con mecanismos ligeramente diferentes en el acoplamiento del centro. Se incluyen figuras que muestran cómo L2-norm, GWD y KLD se comportan frente a variaciones de escala y de parámetros del objeto target (ht).

* **Problemas atacados**: Demostrar que las propiedades de auto-modulación de KLD son robustas a la elección de la forma (Np||Nt vs. Nt||Np).
* **Limitaciones de ese entonces**: La asimetría de KLD podía ser vista como una limitación; el análisis de la forma inversa demuestra que ambas formas tienen propiedades similares.
* **Soluciones alcanzadas**: El análisis completo de gradientes para ambas formas confirma que la auto-modulación adaptativa es una propiedad fundamental de KLD y no un artefacto de la dirección de la divergencia.
