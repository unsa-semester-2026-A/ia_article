# ODGEN: Domain-specific Object Detection Data Generation with Diffusion Models

- **Key**: ZhuJ2024
- **Year**: 2024
- **Venue**: NeurIPS

## Section-by-Section Analysis

### Introduction
* **Problems Addressed**: Data scarcity for object detection models (YOLOv5/v7) and limitations in generating multi-class objects and dense scenes with occlusions.
* **Limitations at the time**: Large-scale models trained on web crawl data (LAION) have a domain gap with specific domains (medical, video games). Furthermore, multi-class prompts lead to "concept bleeding" (unintended merging of distinct visual elements), overlapping objects are merged, and some objects are neglected.
* **Solutions Achieved**: Introduced ODGEN, which fine-tunes a pre-trained diffusion model on target distributions and employs object-wise conditioning with spatial constraints and textual descriptions to control the generation.

### Related Works
* **Problems Addressed**: The need for spatial control in layout-to-image generation and dataset synthesis for detector training.
* **Limitations at the time**: GLIGEN struggles to generalize to specific domains as it only inserts gated self-attention layers. ReCo and GeoDiffusion require abundant data to learn layout encoding in text prompts. ControlNet compresses all text descriptions into a global prompt, causing concept bleeding. Paste-based synthesis shows artifacts.
* **Solutions Achieved**: A novel object-wise conditioning method addressing multi-class objects, occlusions, and specific domains simultaneously.

### Method
* **Problems Addressed**: Generating high-fidelity specific domain data and accurate layout control without concept bleeding.
* **Limitations at the time**: Fine-tuning only on full images loses fine-grained object textures. Global text encoding merges multiple categories.
* **Solutions Achieved**: 
  1) **Domain-specific fine-tuning**: Training jointly on cropped foreground patches and entire images.
  2) **Object-wise conditioning**: Uses a *text list* (encoding each class separately with CLIP) and an *image list* (pasting synthetic patches on an empty canvas as spatial guidance for ControlNet) to prevent interference.
  3) **Dataset synthesis pipeline**: Filtering corrupted generated labels using a ResNet50 Foreground/Background discriminator.

### Experiments
* **Problems Addressed**: Validation of the fidelity (FID) and trainability (mAP) of the synthetic data in both specific and general domains.
* **Limitations at the time**: Existing models like ReCo or GeoDiffusion fail to generalize with limited data (e.g., 200 images), and ControlNet struggles with complex object generation.
* **Solutions Achieved**: Evaluated on 7 domain-specific datasets (Roboflow-100 subsets) and COCO-2014. Demonstrated that adding synthetic data improves YOLOv5s/YOLOv7 performance significantly (up to 25.3% mAP). FID scores also consistently outperformed all baselines.

## Overall Synthesis

### Comparative Analysis
ODGEN is compared against several layout-to-image models: ReCo, GLIGEN, ControlNet, GeoDiffusion, InstanceDiffusion, and MIGC. 
- **GLIGEN** fails to generalize to highly specific domains (e.g., MRI, Apex Game) because it only updates inserted self-attention layers.
- **ReCo and GeoDiffusion** integrate bounding boxes into text tokens but require large amounts of data to align modalities, failing when fine-tuned with only 200 images.
- **ControlNet** integrates bounding box information via visual conditions, but still suffers from concept bleeding, missing objects, or generating incorrect categories in complex dense scenes.
ODGEN significantly outperforms these baselines in visual fidelity (lowest FID) and downstream object detection accuracy.

### State of the Art (SOTA)
ODGEN advances the SOTA in dataset synthesis for object detection, particularly for specific domains with scarce data (e.g., 200 real images). It achieves up to a **25.3% mAP@.50:.95** improvement over the baseline on YOLO detectors (YOLOv5s and YOLOv7) across 7 specialized domains (e.g., underwater, thermal/medical, gaming), outperforming GLIGEN, ReCo, ControlNet, and GeoDiffusion. In general domains (COCO-2014), it provides up to a **5.6% mAP@.50:.95** advantage.

### Methodology
ODGEN utilizes a three-stage approach:
1. **Domain-specific Diffusion Fine-tuning**: Fine-tunes Stable Diffusion (v2.1) using both entire scenes and cropped foreground objects (resized to 512x512) to learn specific domain styles and detailed object textures.
2. **Object-wise Conditioning (ControlNet)**: 
   - **Text List Encoding**: Encodes the class name of each object individually using a frozen CLIP text encoder, followed by a trainable text embedding encoder, avoiding concept bleeding.
   - **Image List Encoding**: Uses the fine-tuned diffusion model to generate single-object patches, which are resized and pasted onto an empty canvas based on bounding box coordinates, providing explicit spatial guidance without occlusion overlap.
3. **Synthesis Pipeline & Filtering**: Samples bounding box attributes (number, area, aspect ratio, location) from multivariate Gaussian distributions based on the training set. A fine-tuned ResNet50 discriminator acts as a quality filter to discard pseudo labels where foreground objects failed to generate.

### Metrics
- **Performance / Trainability**: Evaluated on YOLOv5s and YOLOv7. In the low-data regime (200 real images + 5000 synthetic images), ODGEN improved mAP@.50:.95 by up to **25.3%** on 7 Roboflow-100 datasets (Apex Game, Robomaster, MRI Image, Cotton, Road Traffic, Aquarium, Underwater). On COCO-2014, it improved mAP by up to **5.6%**.
- **Fidelity**: Achieved the lowest **FID** scores compared to baselines on all 7 evaluated domain-specific datasets (e.g., 58.21 on Apex Game, 93.82 on MRI, 70.20 on Underwater).
- **Hyperparameters**: Fine-tuning for 3,000 iterations; ControlNet trained for 200 epochs. Reconstruction loss used foreground re-weighting ($\gamma = 25$).
