# Estado del Arte de los Modelos de Difusión en Visión Computacional

- **Tema**: Modelos de Difusión, Visión de Bajo Nivel, Edición de Imágenes, Aumentación Científica y Desafíos Éticos.
- **Fecha**: 12 de julio de 2026

## 1. Idea Central
Los modelos de difusión basados en eliminación de ruido (Denoising Diffusion Models) se han consolidado como el estándar de oro para tareas generativas y de restauración en visión artificial. Sin embargo, su despliegue práctico enfrenta desafíos críticos en velocidad de cómputo, fidelidad a restricciones físicas (en el dominio científico) y vulnerabilidades de privacidad. Paralelamente, surgen técnicas eficientes de adaptación zero-shot y arquitecturas alternativas como los Modelos de Espacio de Estados (SSMs).

---

## 2. Desarrollos Clave y Tareas Abarcadas

### A. Visión de Bajo Nivel e Inpainting
Los modelos de difusión superan a las GANs y VAEs tradicionales en tareas de restauración de bajo nivel (super-resolución, eliminación de ruido e inpainting), generando texturas finas con mayor realismo perceptivo mediante procesos continuos o discretos de difusión inversa [[He2025]]. En inpainting de imagen y video, los DMs facilitan la síntesis de contenido semánticamente coherente con el entorno de la máscara [[Quan2024]].

### B. Edición de Imágenes y Aumentación de Datos
Los modelos latentes de difusión (LDMs) controlados por texto, bocetos o imágenes de referencia permiten una edición semántica precisa (local y global) sin alterar regiones no objetivo [[Huang2025]]. Además, actúan como potentes herramientas de aumentación de datos, permitiendo inyectar variabilidad semántica, adaptar dominios y personalizar sujetos de entrenamiento para mejorar clasificadores y detectores downstream [[Alimisis2025Advances]].

### C. Adaptación Zero-Shot para Análisis Denso (Vision Geométrica)
Se ha descubierto que los LDMs preentrenados con miles de millones de imágenes de internet poseen un entendimiento geométrico tridimensional implícito del mundo físico. Métodos como Marigold [[Ke2025Marigold]] demuestran que es posible adaptar estos modelos para predecir profundidad monocular y normales de superficie con un ajuste fino mínimo (pocos días en una sola GPU de consumo) obteniendo una generalización zero-shot sobresaliente en escenas invisibles.

### D. Limitaciones de Coherencia Física en Ciencia
Aunque los DMs sobresalen en la generación de imágenes naturales hermosas, presentan severas deficiencias en la generación de imágenes científicas (como tomografías de rocas porosas o botánica). El estudio comparativo de [[Sordo2025Synthetic]] demuestra que modelos como StyleGAN (GAN) preservan mucho mejor las restricciones topológicas y la conectividad física de los materiales, mientras que los DMs comerciales tienden a introducir artefactos que violan leyes físicas.

### E. Desafíos de Privacidad y Seguridad
La adopción masiva de DMs expone riesgos de fugas de datos de entrenamiento (data leakage), infracciones de derechos de autor y generación de desinformación (deepfakes). Para contrarrestar esto, se desarrollan técnicas de envenenamiento de datos preventivo (para evitar que artistas sean plagiados) y algoritmos de borrado de conceptos (concept erasing) [[Chen2025Comprehensive]].

### F. Modelos de Espacio de Estados (SSMs) como Alternativa Eficiente
Para combatir la lentitud de muestreo iterativo de los DMs y la complejidad cuadrática de los Transformers, surgen alternativas basadas en Mamba de complejidad lineal, como VmambaIR [[Shi2024VmambaIR]]. Este enfoque demuestra rendimiento del estado del arte en restauración visual con una fracción del costo en memoria VRAM y parámetros, postulándose como el backbone del futuro para visión en tiempo real en dispositivos edge.

---

## 3. Conclusiones y Síntesis del Estado del Arte

1. **Eficiencia e Inferencia**: La latencia del proceso de muestreo inverso sigue siendo el principal cuello de botella de los DMs. Mientras que enfoques como Marigold aceleran el entrenamiento, el procesamiento en tiempo real se beneficia de optimizaciones arquitectónicas o de alternativas lineales como Mamba.
2. **Exactitud vs. Realismo**: Existe una brecha entre el realismo perceptivo (evaluado con FID/LPIPS) y la exactitud física/científica. Los modelos generativos científicos necesitan restricciones físicas duras integradas directamente en su pérdida.
3. **Priors Generativos como Backbones**: El uso de modelos generativos masivos texto a imagen como estimadores geométricos condicionales representa un cambio de paradigma, reduciendo drásticamente la dependencia de datos reales anotados con sensores LiDAR.

---

## 4. Enlaces de Literatura Relacionados
- [[He2025]]
- [[Quan2024]]
- [[Alimisis2025Advances]]
- [[Sordo2025Synthetic]]
- [[Huang2025]]
- [[Ke2025Marigold]]
- [[Chen2025Comprehensive]]
- [[Shi2024VmambaIR]]
