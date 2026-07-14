# Notes: Fatima 2025 - Corner Cases: How Size and Position of Objects Challenge ImageNet-Trained Models

## 1. Section-by-Section Analysis

### Introduction
- Highlights the vulnerability of Deep Neural Networks (DNNs) to spurious correlations (e.g., predicting "sea lion" based on water).
- Argues that dataset biases, specifically positional bias (human preference to center objects) and size bias (making objects large), cause models to rely more on spurious features when objects are small and decentralized.
- Mentions that current mitigation strategies like retraining the last layer (Kirichenko et al., 2023) or distributionally robust optimization (Sagawa et al., 2020) assume constant object sizes and locations, which fails when models are deployed in the wild.

### Related Work
- Discusses spurious features, existing datasets (Waterbirds, ImageNet-9, RIVAL10, Spawrious), and biases in image datasets.
- Identifies a gap in the literature: previous datasets and methods lack fine-grained, simultaneous control over object size, spatial location, and their interactions with background spuriousity levels.

### Biases in ImageNet
- Proposes quantitative metrics for positional and size biases:
  - **Centeredness Score ($C_c$):** Based on the $L_\infty$ distance between the image center and object center. The average score across classes is $\approx 0.747$.
  - **Size Score ($S_c$):** Ratio of object height/width to image height/width. The average score is $\approx 0.417$.
- Correlates these scores with spuriousity levels by evaluating ConvNeXt-Base on inpainted ImageNet images. Finds a negative correlation between spurious feature reliance and the product of size and center scores (Kendall's $\tau = -0.293$, Spearman's $\rho = -0.416$).

### Hard-Spurious-ImageNet
- Introduces a synthetic dataset where core objects are cropped via bounding boxes and the original backgrounds are inpainted using Segment Anything (SAM) and LaMa.
- Objects are resized to three resolutions: 56x56, 84x84, and 112x112 pixels (on a 224x224 canvas).
- The dataset features four distinct groups for evaluation based on object location and background:
  - **CeO:** Center location, Original background
  - **CoO:** Corner (top-right) location, Original background
  - **CeR:** Center location, Random background
  - **CoR:** Corner (top-right) location, Random background
- Further introduces **Hard-Spurious-ImageNet-10** (10 classes highly reliant on core features paired with 10 highly spurious backgrounds) and an **Aspect-Ratio (AR) preserving variant**.

### Experimental Results
- Evaluates various architectures (ResNet-50, ConvNeXt-Base, CoAtNet, Hiera, MViTv2, CLIP, EVA-CLIP).
- Performance drops precipitously on the "Hard" groups (CoR with small sizes). For instance, ResNet-50's clean accuracy is 81.21%, but it plummets to 3.28% on the 56x56 CoR original group.
- Adding standard data augmentations (MixUp, CutMix, AutoAugment) improves overall accuracy but paradoxically harms performance on the hardest group (CoR accuracy drops from 3.28% to 2.93%).
- Group robustness approaches (JTT, DFR) and standard ERM are tested on a balanced training subset. DFR achieves the highest Hard group accuracy (59.79%) but slightly trails behind standard ERM on the Easy group (72.47% vs. 74.84%). JTT struggles to generalize, lowering performance across the board (Hard group at 46.49%).

### Challenges and Future Work
- **Limitations:** The pipeline relies on bounding boxes rather than perfect segmentation masks, occasionally leaving core features in the background. Secondary background objects and clutter also make learning small core features harder.
- **Future Work:** Evaluating non-fixed random object locations, experimenting with other network architectures (like contrastive learning), and expanding to other datasets.

### Conclusion
- Hard-Spurious-ImageNet successfully reveals the significant vulnerabilities of models when objects are small and decentralized.
- Current group robustness methods remain insufficient when spurious correlations are entangled with scale and positional biases.

## 2. Overall Synthesis

### Comparative Analysis
- Unlike datasets such as Waterbirds or ImageNet-9, Hard-Spurious-ImageNet isolates and tests the specific geometric properties of core features (size and placement) against spurious backgrounds.
- Vision-Language Models like EVA-CLIP drastically outperform pure Vision Transformers and CNNs in these corner cases. Furthermore, scaling up model sizes (e.g., ViT-Tiny to ViT-Large) yields consistent improvements across all groups (e.g., CoR 56x56 jumps from 1.83% to 11.09%).
- Standard debiasing methods fail to scale to geometric variations. Deep Feature Reweighting (DFR) manages a slight edge on the worst groups over Empirical Risk Minimization (ERM) (59.79% vs 57.56% on Hard), but methods like Just Train Twice (JTT) perform poorly.
- ERM trained with a balanced distribution of groups ($ERM_{all}$) narrows the performance gap between Easy, Medium, and Hard cases much better than ERM trained primarily on Easy data ($ERM_{easy}$).

### SOTA (State of the Art) Performance
- **EVA-CLIP (Vision-Language Model):** Achieves the best overall robustness on this benchmark. Even in the harshest condition (CoR, 56x56), it achieves 26.79% accuracy on the original dataset and 45.70% on the AR-preserving dataset. This greatly surpasses standard models like ConvNeXt-Base (7.12% / 18.42%).
- **AR-Preservation:** Models universally perform better on the aspect-ratio-preserving variant of the dataset compared to the original fixed-box dataset, emphasizing that resolution-warping artifacts exacerbate spurious feature reliance.

### Methodology
- **Data Generation:** 
  1. Core features (objects) are localized using ImageNet-1K ground truth bounding boxes.
  2. The object is removed, and the remaining background is realistically completed using Inpaint-Anything and the LaMa inpainting model.
  3. The cropped object is resized to target resolutions (56x56, 84x84, 112x112) representing 1/16th, 9/64ths, and 1/4th of the 224x224 image area.
  4. The object is superimposed onto either the original inpainted background or a random inpainted background, at either the center or the top-right corner.
- **Subgroup Categorization:**
  - **Easy:** CeO and CoO (84x84 and 112x112)
  - **Medium:** CeO and CoO (56x56), CeR and CoR (112x112)
  - **Hard:** CeR and CoR (56x56 and 84x84)

### Metrics
- **Center Score ($C_c$):** $1 - \frac{1}{MN} \sum_{i,j} ||I_{i,c} - O_{i,j,c}||_\infty$ (measures positional bias).
- **Size Score ($S_c$):** $\frac{1}{MN} \sum_{i,j} \frac{h_j w_j}{H_i W_i}$ (measures scale bias).
- **Subgroup/Group Accuracies:** Evaluated individually across CeO, CoO, CeR, and CoR categories.
- **Aggregated Performance Profiles:** Metrics grouped by Easy, Medium, and Hard conditions.
- **Statistical Correlation:** Kendall’s $\tau$ and Spearman’s $\rho$ used to compute the inverse relationship between spuriousness (accuracy of models on background-only inpainted images) and positional/size biases.
