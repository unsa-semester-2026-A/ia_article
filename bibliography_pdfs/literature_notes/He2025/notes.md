# Diffusion Models in Low-Level Vision: A Survey
- **Key**: He2025
- **Year**: 2025
- **Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

# 1. Section-by-Section Analysis

## Abstract & Introduction

### Summary
El artículo presenta la primera revisión sistemática de los modelos de difusión de eliminación de ruido aplicados a tareas de visión artificial de bajo nivel. El objetivo principal es estructurar y unificar teóricamente los avances metodológicos y de aplicación práctica para facilitar futuras investigaciones en este dominio.

### Most Important Data
- **Contexto SOTA**:
  - Modelos generativos anteriores como GANs y VAEs sufren de colapso de modo, inestabilidad durante el entrenamiento y difuminación de texturas finas.
  - Los modelos de difusión (DMs) han emergido como el nuevo estándar de oro por su estabilidad matemática y su capacidad para sintetizar detalles de textura realistas.
- **Objetivos de la Revisión**:
  - Unificar las tres formulaciones teóricas generales de DMs (DDPM, SDE, SGMs).
  - Proporcionar una taxonomía de bajo nivel cubriendo imágenes naturales, médicas, satelitales y video.
  - Evaluar cuantitativamente los modelos sobre seis tareas comunes con métricas estandarizadas.

---

## Theoretical Foundations of Diffusion Models

### Summary
Esta sección detalla los fundamentos matemáticos de las tres principales familias de modelos de difusión y establece su conexión matemática. Asimismo, se describen sus relaciones teóricas con otros modelos generativos como VAEs, GANs y Flujos Normalizadores.

### Most Important Data
- **DDPM (Denoising Diffusion Probabilistic Models)**:
  - Proceso de difusión directa discretizado que añade ruido gaussiano:
    $$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$
  - Proceso inverso parametrizado por una red neuronal que estima el ruido agregado $\epsilon_\theta(x_t, t)$.
- **SDE (Stochastic Differential Equations)**:
  - Modelado en tiempo continuo donde la adición y eliminación de ruido se rigen por ecuaciones diferenciales estocásticas hacia adelante e inversas:
    $$dx = f(x,t)dt + g(t)dw$$
- **SGMs (Score-based Generative Models)**:
  - Generación mediante estimación del gradiente de la densidad logarítmica de los datos, denominado score matching:
    $$\nabla_x \log p_t(x)$$
- **Conexiones**:
  - DDPM se demuestra como un caso discretizado de SDE (Variance Preserving SDE), y SGM corresponde a otra variante de SDE (Variance Exploding SDE).

---

## Taxonomy of Low-Level Vision Applications

### Summary
Se introduce una taxonomía integral para organizar los métodos basados en difusión aplicados a visión de bajo nivel. Clasifica la literatura según el espacio de representación (píxel vs. latente), las estrategias de acondicionamiento y el dominio visual.

### Most Important Data
- **Espacio de Difusión**:
  - Pixel-space: Difusión aplicada directamente sobre los píxeles (alta fidelidad espacial, pero de coste computacional extremo).
  - Latent-space (LDMs): Difusión en un espacio latente de menor dimensión codificado por un autoencoder (gran ahorro de recursos y mejor extracción semántica).
- **Acondicionamiento**:
  - Concatenación directa de la imagen degradada a la entrada del U-Net.
  - Mecanismos de atención cruzada (cross-attention) y adaptadores de red (ControlNet) para guiar la reconstrucción con guías externas (texto, mapas de profundidad).
- **Dominios de Aplicación**:
  - Imágenes naturales, imágenes médicas (CT, MRI), teledetección (UAV y satelital) y restauración de video.

---

## Benchmarks & Quantitative Evaluation

### Summary
Esta sección presenta una evaluación comparativa sistemática de los métodos de difusión frente a los baselines tradicionales en seis tareas comunes. Utiliza métricas estandarizadas para medir tanto la fidelidad geométrica como la calidad perceptiva visual.

### Most Important Data
- **Métricas de Evaluación**:
  - Fidelidad de pixeles: PSNR (Peak Signal-to-Noise Ratio) y SSIM (Structural Similarity).
  - Calidad perceptiva: FID (Fréchet Inception Distance) y LPIPS (Learned Perceptual Image Patch Similarity).
- **Resultados de Rendimiento**:
  - En super-resolución de rostros, los DMs logran puntuaciones de LPIPS hasta un **25% menores** (mejor calidad perceptiva) que las CNNs deterministas.
  - Sin embargo, los DMs obtienen consistentemente peores resultados en PSNR/SSIM debido a la naturaleza estocástica del muestreo, que alucina detalles realistas pero geométricamente distintos del original exacto.

---

## Challenges and Future Directions

### Summary
Se analizan los cuellos de botella actuales de los modelos de difusión y se proyectan cuatro rutas de investigación prioritarias. El enfoque principal es la reducción de costes de cómputo para permitir aplicaciones interactivas en tiempo real.

### Most Important Data
- **Desafíos Críticos**:
  - Alta latencia: La necesidad de múltiples pasos de inferencia inversa (típicamente de 50 a 1000 pasos en DDPM clásicos) hace que el procesamiento de video en tiempo real sea impracticable.
  - Requisitos de VRAM elevados.
- **Rutas de Investigación**:
  - Aceleración del muestreo inverso mediante solvers de ecuaciones diferenciales ordinarias (ODE solvers como DPMSolver) y destilación de conocimiento (Knowledge Distillation).
  - Integración de restricciones geométricas de consistencia física para evitar alucinaciones erróneas.
  - Desarrollo de métricas específicas para medir la coherencia física en imágenes científicas.

---

# 2. Overall Synthesis & Comparative Analysis

| Metric / Component | State of the Art Context (GANs, CNNs, Transformers) | Denoising Diffusion Models (Surveyed in He2025) |
|--------------------|----------------------------------------------------|------------------------------------------------|
| **Methodology / Features** | Optimización supervisada directa (L1/L2 loss), entrenamiento adversarial inestable (GANs), campos receptivos locales o globales con autoatención. | Proceso difusivo hacia adelante (adición de ruido gaussiano), proceso inverso de eliminación de ruido iterativo, modelado en espacio latente (LDM). |
| **Perceptual Metrics (LPIPS/FID)** | Tienden a producir artefactos de alta frecuencia (GANs) o imágenes difuminadas sin detalles de textura (CNNs/Transformers con pérdidas L1/L2). | Excelente calidad de texturas finas, FID y LPIPS notablemente mejores (más bajos) que los enfoques deterministas clásicos. |
| **Pixel Fidelity (PSNR/SSIM)** | Alta correlación de píxeles y excelente PSNR/SSIM por optimización directa de distancias Euclidianas. | PSNR/SSIM subóptimo debido a la naturaleza estocástica de generación que genera contenido plausible pero no pixel-identical. |
| **Inference Latency** | Extremadamente rápida (inferencia en un solo paso forward, típicamente <50ms en GPU). | Lenta (requiere decenas o cientos de pasos iterativos en GPU, típicamente >1s por imagen sin aceleración). |

## Main Research Conclusions
La revisión en He2025 demuestra que los modelos de difusión han redefinido los límites de la calidad perceptiva en la visión de bajo nivel, superando las limitaciones tradicionales de entrenamiento de las GANs. El principal desafío restante para su despliegue comercial o en dispositivos de borde (edge) radica en la latencia de inferencia, impulsando investigaciones en destilación de pasos de muestreo y arquitecturas híbridas eficientes.
