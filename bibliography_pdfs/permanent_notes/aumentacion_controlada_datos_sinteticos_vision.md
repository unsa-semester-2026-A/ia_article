# Aumentación Controlada y Datos Sintéticos en Modelos de Visión Computacional

- **Tema**: Aumentación de Datos Generativa, Modelos de Difusión Controlables (ControlNet), Fusión Fotorrealista, Robustez de Grupo y Datos Sintéticos en Detección/Segmentación.
- **Fecha**: 13 de julio de 2026

## 1. Idea Central
La generación de datos sintéticos para tareas de visión espacialmente dependientes (detección, segmentación, visual grounding) ha madurado desde la simple síntesis incondicional o copy-paste clásico hacia pipelines generativos altamente controlables. Los desafíos tradicionales de desalineación espacial, omisión de objetos y "concept bleeding" se resuelven mediante técnicas de **rectificación en tiempo de inferencia**, **condicionamiento a nivel de objeto** y **composición geométrica 3D**. A la par, el inpainting fotorrealista (LaMa) actúa no solo como restaurador de píxeles, sino como una herramienta activa para diagnosticar sesgos de fondo o eliminar oclusiones en tareas cognitivas de alto nivel.

---

## 2. Desarrollos Clave y Tareas Abarcadas

### A. Aislamiento de Sujeto y Diversificación de Fondo (Background Augmentation)
Para evitar que los clasificadores y detectores se sobreajusten a características espurias del fondo o sufran por corrupción del sujeto en la síntesis de difusión, surgen arquitecturas de aislamiento:
* **AGA (Automated Generative Data Augmentation) [[Rahat2024]]**: Propone segmentar el sujeto original (usando SAM y GroundingDINO guiado por nombres de superclases para robustecer la localización). Una vez aislado, el sujeto es sometido a transformaciones afines y re-colocado en fondos diversificados generados por Stable Diffusion, guiados por descripciones de Llama mediante descomposición de prompts. Esto produce variaciones realistas que mejoran la precisión in-distribution (+15.6%) y out-of-distribution (+23.5%).
* **CIA (Controllable Image Augmentation) [[Benkedadra2024]]**: Introduce un pipeline modular que extrae características de control (Canny, segmentación, poses) y condiciona la síntesis mediante ControlNet, modificando el prompt con vocabularios de palabras permutadas. Valida que Canny y OpenPose son óptimos para aumentación en YOLOv8, mientras que extractores de landmarks densos (MediaPipe) degradan el mAP al inducir desalineaciones de bounding box.

### B. Condicionamiento Espacial por Región de Inferencia (Training-Free & ControlNet)
Garantizar que los objetos generados coincidan exactamente con las coordenadas de la caja delimitadora (bounding box) sin incurrir en entrenamientos costosos es resuelto mediante:
* **ReCon [[ZhuH2025]]**: Diseña dos mecanismos en tiempo de inferencia: *Region-Guided Rectification (RGR)*, que detecta discrepancias de IoU durante el sampling y reinyecta latentes ruidosas corregidas en etapas tempranas (0.75T), y *Region-Aligned Cross-Attention (RACA)*, que limita el alcance de los mapas de atención cruzada de cada clase a sus bounding boxes respectivas. Es un framework plug-and-play libre de reentrenamiento.
* **ODGEN [[ZhuJ2024]]**: Aborda el concept bleeding en escenas densas mediante *Object-wise Conditioning*. Fine-tunea el UNet en imágenes completas y crops de objetos del dominio de interés, codificando la clase e imagen espacial de cada objeto de manera independiente (mediante una lista de textos e imágenes que se concatenan como guía en ControlNet). Mejora el mAP hasta en un +25.3% en YOLOv5/v7.

### C. Composición Geométrica 3D y Armonización Generativa
Superar la artificialidad de los bordes del copy-paste y las distorsiones de perspectiva se logra coordinando la profundidad física y la mezcla de color:
* **SOC (Synthetic Object Compositions) [[HuangW2025]]**: Genera 20M de parches de objetos organizados por categorías y los proyecta sobre escenas mediante una aumentación de disposición 3D (bounding box 3D y planos de piso) y desenfoque por profundidad (defocus blur). Para garantizar el realismo visual, aplica armonización generativa mediante una mezcla Lab-space reponderada por el área de la máscara. Modelos entrenados con 100K imágenes SOC superan a los entrenados con datasets reales masivos como V3Det o GRIT.

### D. Inpainting como Herramienta de Diagnóstico y Oclusión
El inpainting basado en Fast Fourier Convolutions (LaMa) se consolida como una herramienta de preprocesamiento de alta fidelidad:
* **Cuantificación de Spuriosity [[Fatima2025]]**: Utiliza inpainting con LaMa en el framework *Inpaint-Anything* para borrar selectivamente el core object de ImageNet-1k. La precisión del clasificador sobre estas imágenes vacías define el nivel de "spuriosity" (dependencia del fondo) de cada clase. El benchmark resultante, *Hard-Spurious-ImageNet*, demuestra que los clasificadores fallan catastróficamente ante objetos pequeños y descentrados en fondos espurios (grupo CoR).
* **Scene Reconstruction en SOR [[Liu2026]]**: Propone *PairwiseSOR-MLMs* para salient object ranking. Al descomponer el ranking en comparaciones de pares, utiliza inpainting con LaMa para eliminar objetos ocluyentes y reconstruir el fondo del escena. Esto permite a los modelos multimodales (MLMs) evaluar la importancia visual de cada objeto por separado sin la interferencia del ruido visual de oclusión.

### E. Filtrado por Active Learning sobre el Pool Sintético
* **AVM Parking-Slot Detection [[Zhang2026]]**: Advierte que la generación masiva de imágenes sintéticas sin criterio induce una saturación de información y puede degradar la generalización. Propone una estrategia basada en Active Learning para calcular la entropía del detector en el pool sintético y seleccionar únicamente las muestras difíciles (mayor incertidumbre). La mezcla de 40% datos reales y 60% sintéticos filtrados maximiza la precisión en sistemas Surround View (AVM).

---

## 3. Conclusiones y Síntesis del Estado del Arte

1. **Alineación Geométrica Estricta**: La aumentación generativa para detección/segmentación es inviable si no se imponen restricciones espaciales fuertes en la inferencia (como RGR en ReCon o la lista de imágenes en ODGEN) para evitar el desplazamiento de los objetos fuera de sus pseudo-labels.
2. **Coherencia Física 3D**: La armonización de iluminación y el desenfoque por profundidad (defocus blur) disminuyen drásticamente el gap sim-to-real, permitiendo que la composición sintética supere en generalización al copy-paste clásico y a la difusión incondicional.
3. **Filtro de Utilidad frente a Filtrado de Calidad**: La utilidad de una imagen generada para entrenar un modelo downstream no se correlaciona con su calidad estética visual (evaluada por métricas IQA como NIMA o BRISQUE), sino con su dificultad y diversidad semántica, lo cual se explota mediante Active Learning sobre el pool sintético.

---

## 4. Enlaces de Literatura Relacionados
- [[Rahat2024]]
- [[Fatima2025]]
- [[Liu2026]]
- [[Zhang2026]]
- [[Benkedadra2024]]
- [[ZhuJ2024]]
- [[HuangW2025]]
- [[ZhuH2025]]
