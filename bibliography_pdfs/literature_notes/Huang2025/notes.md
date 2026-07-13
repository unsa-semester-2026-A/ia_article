# Diffusion Model-Based Image Editing: A Survey
- **Key**: Huang2025
- **Year**: 2025
- **Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

# 1. Section-by-Section Analysis

## Abstract & Introduction

### Summary
Este survey exhaustivo aborda el estado del arte en la edición de imágenes basada en modelos de difusión. Su objetivo es clasificar y analizar críticamente la literatura científica del sector, cubriendo estrategias de aprendizaje, tipos de acondicionamiento e introduciendo un nuevo benchmark de evaluación.

### Most Important Data
- **El Problema**: Modificar imágenes de forma precisa respetando la coherencia espacial y textural de la escena original.
- **SOTA Issues**:
  - Métodos basados en GANs sufren por un espacio de latentes estrecho que dificulta la inversión perfecta de imágenes reales (reconstruction leakage).
- **Propuesta**:
  - Taxonomía unificada para algoritmos de edición por difusión.
  - Benchmark EditEval para evaluar cuantitativamente la edición de texto a imagen.

---

## Categorization of Diffusion-Based Editing

### Summary
Presenta una clasificación detallada de las tecnologías de edición de imágenes por difusión. Clasifica los enfoques según si requieren ajuste fino de parámetros (LoRA, adaptadores) o si son libres de entrenamiento mediante manipulación de mapas de atención.

### Most Important Data
- **Clasificación por Estrategia**:
  - **Técnicas Tuning-free**: Modifican los mapas de cross-attention (ej. Prompt-to-Prompt) durante la inferencia para guiar la generación sin alterar la arquitectura o pesos.
  - **Técnicas Tuning-based**: Ajustan los pesos de la red o entrenan pequeños módulos adaptadores (ControlNet, LoRA) para inyectar condiciones espaciales específicas (bordes, poses).
- **Tipos de Acondicionamiento**:
  - Instrucciones de texto (natural language).
  - Guías visuales (máscaras de inpainting, bocetos, mapas de normales).

---

## Inpainting and Outpainting Methodologies

### Summary
Revisa las metodologías específicas dedicadas a rellenar regiones internas de imágenes (inpainting) o extender las fronteras espaciales (outpainting) usando modelos de difusión. Compara enfoques tradicionales guiados por contexto y modernos guiados por texto.

### Most Important Data
- **Mecanismos de Inpainting**:
  - Difusión condicional sobre la frontera de la máscara: El proceso inverso mezcla el ruido generado en la zona vacía con la información real de los píxeles adyacentes no enmascarados en cada paso temporal.
- **Outpainting**:
  - Extensión infinita de bordes mediante el desplazamiento lateral de parches con solapamiento y la unificación de predicciones utilizando mecanismos de atención global para evitar costuras visuales.

---

## EditEval Benchmark and Evaluation Metrics

### Summary
Esta sección introduce el benchmark EditEval para estandarizar la evaluación de algoritmos de edición guiada por texto. Propone una métrica automatizada robusta basada en Modelos de Lenguaje y Visión (LMMs) para correlacionar con el juicio humano.

### Most Important Data
- **EditEval**:
  - Conjunto de datos curado con múltiples pares de imágenes, instrucciones de texto de edición y ground truths de referencia.
- **Métrica LMM Score**:
  - Utiliza modelos grandes multimodal (como GPT-4V o LLaVA) configurados como evaluadores objetivos para juzgar si la imagen modificada conserva la identidad del sujeto y cumple fielmente la directiva de texto.
  - Supera la correlación humana en comparación con métricas tradicionales como la distancia CLIP de texto a imagen.

---

## Future Directions

### Summary
Identifica los desafíos técnicos que limitan la edición interactiva en tiempo real y propone soluciones futuras. Se enfoca en resolver las fugas de color (leakage) y el alto coste computacional.

### Most Important Data
- **Desafíos Técnicos**:
  - **Leakage (Fuga de edición)**: Modificación no intencionada de píxeles fuera de la región objetivo.
  - Tiempos de procesamiento altos: Inviabilidad de la edición interactiva instantánea en navegadores web o dispositivos móviles debido al muestreo multi-paso.
- **Rutas de Investigación**:
  - Edición en un solo paso temporal (One-step Diffusion Editing) utilizando modelos destilados por consistencia (Consistency Models).
  - Máscaras de atención auto-limitadas para aislar completamente la propagación de gradientes de edición.

---

# 2. Overall Synthesis & Comparative Analysis

| Metric / Component | State of the Art Context (GAN-based Image Editing) | Diffusion-Based Image Editing (Surveyed in Huang2025) |
|--------------------|----------------------------------------------------|------------------------------------------------------|
| **Methodology / Features** | Inversión del espacio latente de GANs ($w/w^+$ space), limitada a dominios muy estrechos (ej. caras). | Manipulación de mapas de atención cruzada (cross-attention), inyección de priors latentes de Stable Diffusion. |
| **Leakage Control** | Falla severamente: cambiar la expresión de una cara a menudo altera la identidad de la persona o el fondo. | Control preciso mediante retención de mapas de autoatención de la imagen original (Prompt-to-Prompt). |
| **Input Modalities** | Principalmente limitado a vectores latentes e interfaces de control rígidas. | Multimodal: Texto, bocetos, máscaras densas, mapas de normales y profundidad. |
| **Evaluation Method** | Métricas manuales subjetivas o CLIPScore estadístico. | Benchmark **EditEval** utilizando **LMM Score** para alineación perfecta con el juicio humano. |

## Main Research Conclusions
La edición de imágenes basada en modelos de difusión ha superado la barrera de los dominios estrechos que limitaba a las GANs, ofreciendo control multimodal sin precedentes. A través de la introducción de EditEval y LMM Score, el campo cuenta ahora con herramientas de evaluación más justas, impulsando la investigación hacia la edición libre de entrenamiento en un solo paso temporal para aplicaciones comerciales en tiempo real.
