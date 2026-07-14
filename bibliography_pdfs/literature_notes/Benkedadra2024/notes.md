# Literature Note: CIA: Controllable Image Augmentation Framework Based on Stable Diffusion

**Metadata:**
- **Authors:** Mohamed Benkedadra, Dany Rimez, Tiffanie Godelaine, Natarajan Chidambaram, Hamed Razavi Khosroshahi, Horacio Tellez, Matei Mancas, Benoit Macq, Sidi Ahmed Mahmoudi
- **Year:** 2024
- **Venue:** arXiv (cs.CV)

## Section-by-Section Analysis

### I. Introduction
- **Overview:** Highlights the dependence of deep learning models on large, accurately annotated datasets and the limitations of traditional, content-agnostic data augmentation methods.
- **Core Proposal:** Introduces CIA (Controllable Image Augmentation), a modular framework using Stable Diffusion and ControlNet to augment datasets synthetically by generating new scenarios (weather, styles, backgrounds) while respecting original spatial annotations.

### II. Related Works
- **Context:** Reviews classical data augmentation (geometric transformations) and advanced generative methods (Stable Diffusion, DALL-E).
- **Limitations Identified:** Previous approaches like synthetic copy-pasting introduce foreground-background discontinuities. Prior generative methods lack an end-to-end, reliable pipeline for generative data augmentation with quality assessment.
- **Quality Assessment:** Discusses Image Quality Assessment (IQA) metrics like BRISQUE, NIMA, ClipIQA, and Active Learning metrics for evaluating synthetic data impact.

### III. Proposed CIA Framework
- **Architecture:** Consists of four sequential modules:
  1. **Extraction:** Extracts control features from real images using extractors like Canny edges, OpenPose, or segmentation masks.
  2. **Generation:** Generates synthetic images using Stable Diffusion conditioned by the extracted features and text prompts. Prompts are modified automatically using vocabulary substitution.
  3. **Quality Assessor and Sampler:** Filters out low-quality generated images based on chosen IQA or Active Learning metrics.
  4. **Train and Test:** Facilitates training object detection models on mixed proportions of real and synthetic data.

### IV. Experimental Setup
- **Task:** Human object detection using YOLOv8n (trained for 300 epochs, SGD).
- **Datasets:** Subsets of COCO and Flickr30k focused exclusively on the "PERSON" class (object area 5% to 80%).
- **Baselines:** $D_{250}$ and $D_{500}$ (250 and 500 real images respectively).
- **Experiments:** 
  1. ControlNet Effect (Canny Edge, OpenPose, MediaPipe, Segmentation, False-Segmentation).
  2. Data Augmentation Additivity (comparing low, medium, and high classic YOLOv8 augmentations like $\pm 10$ degrees rotation, mosaic, mixup, and copy-paste).
  3. Sampling with Quality Metrics (filtering synthetic sets from $D_{1250}'$).

### V. Results
- **ControlNet Efficacy:** Canny Edge, OpenPose, and Segmentation positively impacted mAP, with improvements peaking at an addition of 750 synthetic samples. MediaPipe and False-Segmentation degraded performance due to object-bounding box misalignment.
- **Additivity:** CIA images proved complementary to classical augmentations. While high levels of classical augmentation caused overfitting, adding CIA-generated images maintained and improved performance across all classical augmentation tiers.
- **Sampling Strategy:** Advanced sampling strategies (ClipIQA, NIMA, BRISQUE, CORE-SET, confidence) did not significantly outperform random sampling.

### VI. Discussion & VII. Conclusion
- **Discussion:** Demonstrates CIA's capability to introduce robust variations (backgrounds, points of view, and styles like photography vs. painting) effectively acting as an advanced regularizer.
- **Conclusion:** Concludes that CIA is a highly adaptable, plug-and-play tool for data-constrained environments, easily extendable to other vision tasks (classification, segmentation).

---

## Overall Synthesis

### State of the Art (SOTA) & Comparative Analysis
Unlike contemporary generative augmentation models that suffer from spatial misalignment or visual discontinuity (e.g., simplistic copy-pasting of DALL-E generated objects), CIA tightly binds the generation process to the original spatial annotations via ControlNet. The methodology proves that intelligently guided diffusion models can closely mimic the performance gain of doubling the real dataset ($D_{500}$) when starting from highly constrained regimes ($D_{250}$). The framework stands out for its modularity and inclusion of built-in quality filtering mechanisms, though empirical evidence showed that random sampling currently remains robust compared to complex IQA metrics.

### Methodology Details
The framework strictly forces the generative output to conform to ground-truth label dimensions by converting the original image into structural maps (e.g., Canny edges, OpenPose skeletons, Segmentation masks). It combines these maps with dynamically modified text prompts using a vocabulary substitution technique (e.g., changing "man in a red shirt" to "woman in a yellow shirt"). The Stable Diffusion v1.5 pipeline processes these inputs to output diverse but spatially consistent fakes, which inherently preserve the original bounding box annotations. 

### Metrics & Quantitative Results
- **Dataset Sizes:** Baseline real images ($D_{250}$, $D_{500}$); synthetic pools up to $D_{1250}'$ (generated by using 5 distinct auto-generated captions for each sample in $D_{250}$).
- **Detection Model:** YOLOv8n, trained for 300 epochs via SGD.
- **Performance:** Adding synthetic images created using Canny Edge or OpenPose significantly increased Mean Average Precision (mAP), approaching the performance of $D_{500}$.
- **Augmentation Additivity:** Evaluated against YOLOv8’s built-in augmentation levels:
  - Low (scale, translation, hue, mosaic)
  - Medium (shear, rotation $\pm 10^\circ$, 10% copy-paste)
  - High (20% copy-paste, mix up)
  The synthetic data mitigated overfitting seen at the "high" classical augmentation level.
- **Quality Filters:** BRISQUE, NIMA, ClipIQA, and model-aware metrics (CORE-SET, low confidence) were evaluated, yet empirical results showed no significant mAP gain over purely random sampling of the generated images.
