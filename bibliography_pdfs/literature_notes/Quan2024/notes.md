# Deep Learning-Based Image and Video Inpainting: A Survey
- **Key**: Quan2024
- **Year**: 2024
- **Venue**: International Journal of Computer Vision (IJCV)

# 1. Section-by-Section Analysis

## Abstract & Introduction

### Summary
El artículo presenta una revisión exhaustiva del inpainting (completado de áreas faltantes) en imágenes y videos mediante aprendizaje profundo. El objetivo es estructurar metodológicamente los avances en tuberías de alto nivel, diseño de módulos y métricas de evaluación perceptiva para guiar las futuras aplicaciones comerciales.

### Most Important Data
- **El Problema**: Rellenar huecos o máscaras en imágenes de forma semántica y texturalmente coherente.
- **Transición del SOTA**:
  - Métodos tradicionales: Copia de parches texturales (PatchMatch) que fallan al reconstruir semántica global compleja.
  - Aprendizaje Profundo: Uso de redes generativas para inferir estructuras complejas de alto nivel a partir del contexto global de la imagen.

---

## High-Level Inpainting Pipelines & Architectures

### Summary
Esta sección categoriza los métodos de inpainting de acuerdo con su flujo de trabajo de alto nivel y sus arquitecturas fundamentales. Compara el uso de redes convolucionales directas, modelos basados en GANs, Transformers autorregresivos y modelos de difusión.

### Most Important Data
- **Estructura de Pipeline**:
  - Una sola etapa (One-stage): Entrada directa de la imagen enmascarada a una red encoder-decoder.
  - Múltiples etapas (Multi-stage/Coarse-to-fine): Una primera red genera una predicción aproximada (baja resolución o contornos) y una segunda etapa refina la textura de alta resolución.
- **Clasificación por Modelo**:
  - CNNs y VAEs: Rápidos pero propensos a resultados borrosos.
  - GANs: Producen detalles nítidos pero sufren inestabilidad en máscaras grandes.
  - Transformers: Excelente modelado de dependencias de largo alcance para inpainting semántico a gran escala.
  - Modelos de Difusión: Capacidad generativa superior y completado invisible mediante condicionamiento de la frontera de la máscara.

---

## Module Design & Attention Mechanisms

### Summary
Analiza las innovaciones en el diseño de bloques constructivos y capas de atención espacial/temporal para guiar el flujo de información contextual hacia las áreas enmascaradas. Detalla la evolución de los tipos de convolución y los mecanismos de autoatención.

### Most Important Data
- **Tipos de Convolución**:
  - Convolución estándar: Trata a los píxeles enmascarados (ceros) como píxeles reales, introduciendo sesgos de color indeseados.
  - Partial Convolution (PConv): Re-escala los valores de la convolución basándose únicamente en la validez de los píxeles no enmascarados.
  - Gated Convolution (GConv): Aprende dinámicamente un mapa de compuerta suave para cada canal, optimizando la transferencia de contexto.
- **Mecanismos de Atención**:
  - Contextual Attention: Busca parches del exterior de la máscara para rellenar el interior.
  - Autoatención 2D (imágenes) y 3D (consistencia temporal en videos).

---

## Datasets, Evaluation Metrics & Applications

### Summary
Describe los conjuntos de datos de prueba, las métricas métricas estándar de evaluación y las principales aplicaciones reales del inpainting. Detalla la brecha entre la evaluación tradicional por píxel y la evaluación perceptiva.

### Most Important Data
- **Datasets Estándar**:
  - Places2, CelebA-HQ (rostros), Paris StreetView.
- **Métricas de Evaluación**:
  - Métricas de píxel (bajas): PSNR, SSIM, MAE (Mean Absolute Error).
  - Métricas perceptivas (altas): FID (Fréchet Inception Distance), LPIPS, IDS (Inpainting Detection Score).
- **Aplicaciones**:
  - Eliminación de objetos indeseados, restauración de películas antiguas dañadas, edición interactiva y eliminación de marcas de agua.

---

## Open Challenges and Future Directions

### Summary
Identifica los problemas no resueltos en el campo y propone líneas de investigación prioritarias. Se enfoca principalmente en la consistencia temporal en videos y el control interactivo por parte del usuario.

### Most Important Data
- **Desafíos Abiertos**:
  - Inpainting de video: Mantener la coherencia temporal libre de parpadeo (jittering) al reconstruir objetos en movimiento.
  - Coherencia estructural en máscaras gigantescas (ultra-large scale inpainting).
  - Fugas de color e incoherencias de frontera.
- **Futuras Direcciones**:
  - Integración de modelos autorregresivos multimodales (texto + imagen).
  - Modelado de flujos ópticos dinámicos para guiar el inpainting temporal en video de forma implícita.

---

# 2. Overall Synthesis & Comparative Analysis

| Metric / Component | State of the Art Context (Traditional & Early DL Inpainting) | Modern Deep Learning Inpainting (Surveyed in Quan2024) |
|--------------------|-------------------------------------------------------------|-------------------------------------------------------|
| **Methodology / Features** | Copia manual de parches (PatchMatch), convoluciones estándar que causan artefactos de color en bordes de máscara. | Convoluciones Gated (GConv), atención contextual dinámica de largo alcance, modelos de difusión condicionales. |
| **Structural Coherence** | Falla completamente en máscaras grandes o en áreas con semántica compleja (ej. caras, textos). | Alta coherencia semántica gracias al uso de priors generativos profundos (DMs, Transformers) y coarse-to-fine. |
| **Temporal Consistency** | No aplicable o produce parpadeos severos (frame-by-frame) en video. | Consistencia temporal mejorada mediante convoluciones 3D y mecanismos de atención espacio-temporal. |
| **Inference Cost** | Rápido pero incapaz de generar contenido nuevo. | Variable: CNNs son rápidas (~10-30ms), mientras que los DMs y Transformers autorregresivos añaden latencia significativa. |

## Main Research Conclusions
El inpainting basado en aprendizaje profundo ha pasado de ser una técnica de corrección de texturas locales a un proceso de síntesis semántica a gran escala. A pesar del gran éxito de los modelos de difusión y transformers en la generación de estructuras complejas, persisten desafíos en el inpainting de video en tiempo real debido a las demandas de coherencia temporal y a la alta latencia computacional de los procesos de muestreo iterativo.
