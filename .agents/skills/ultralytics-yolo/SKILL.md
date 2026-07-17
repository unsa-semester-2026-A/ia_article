---
name: ultralytics-yolo
description: Guides the agent in using Ultralytics YOLO26 for computer vision tasks (detection, segmentation, classification, pose estimation, oriented object detection), including training, validation, inference, custom callbacks, and exporting models.
---

# Ultralytics YOLO26 Agent Skill

This skill provides comprehensive instructions, API summaries, and sitemaps to guide AI agents in leveraging the Ultralytics YOLO26 framework for computer vision applications. 

---

## 1. Overview of Ultralytics YOLO26

Ultralytics YOLO26 is a unified family of real-time computer vision models supporting multiple tasks:
- **Object Detection (`detect`)**: Identifies and localizes objects with bounding boxes.
- **Instance Segmentation (`segment`)**: Identifies object boundaries and outputs instance masks.
- **Semantic Segmentation (`semantic`)**: Performs dense pixel-wise classification of scenes.
- **Pose/Keypoint Estimation (`pose`)**: Detects keypoints (e.g., human joints) in images.
- **Oriented Bounding Boxes (`obb`)**: Detects rotated bounding boxes, ideal for aerial or medical imagery.
- **Classification (`classify`)**: Predicts the category of an entire input image.

### Key Architectural Features
- **Native End-to-End Inference**: Enabled by a default **One-to-One detection head** that outputs direct predictions without requiring Non-Maximum Suppression (NMS) post-processing.
- **Dual-Head Flexibility**:
  - *One-to-One Head (Default)*: No NMS required. Output: `(N, 300, 6)`. Optimized for deployment.
  - *One-to-Many Head*: Standard YOLO output. Output: `(N, nc + 4, 8400)`. Requires NMS, but can offer slightly higher accuracy. Can be enabled via `end2end=False` in predict/val/export modes.
- **DFL-Free Regression**: Distribution Focal Loss is removed in YOLO26, simplifying the head and export process.
- **MuSGD Optimizer**: Hybrid optimizer combining SGD and Muon.
- **Open-Vocabulary (YOLOE-26)**: Supports promptable open-set detection and segmentation using text prompts (e.g., `model.set_classes(["person", "bus"])`) or visual prompts (bounding box examples).

---

## 2. Python API Quick Reference

### Core Usage
```python
from ultralytics import YOLO

# 1. Initialize Model
model = YOLO("yolo26n.pt")      # Load pretrained weights
# model = YOLO("yolo26n.yaml")  # Load model architecture from scratch

# 2. Train Model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device=0)

# 3. Validate Model
metrics = model.val()           # Evaluates on the dataset specified in data config

# 4. Perform Inference (Predict)
results = model.predict(source="path/to/image.jpg", save=True)
# Retrieve predictions
for r in results:
    boxes = r.boxes.xyxy         # Bounding boxes (N, 4)
    scores = r.boxes.conf        # Confidence scores (N, 1)
    clss = r.boxes.cls           # Class indices (N, 1)
    if r.masks is not None:
        masks = r.masks.data     # Segmentation masks (N, H, W)

# 5. Export Model
success = model.export(format="onnx", dynamic=True) # Export format (onnx, engine/TensorRT, openvino, etc.)
```

### Callback System
Register callbacks to execute custom functions at strategic steps:
```python
def on_predict_batch_end(predictor):
    # Access predictor.batch, predictor.results, etc.
    pass

model.add_callback("on_predict_batch_end", on_predict_batch_end)
```

### Custom Trainers
Extend standard training loops by subclassing existing trainers:
```python
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        # Return customized model architecture
        ...
```

---

## 3. Off-line Subdocumentation Map

Detailed, task-specific documentation is available in the `references/` directory. Depending on the user's specific request, you **MUST** read the corresponding subdocumentation file:

1. **High-Level Model Capabilities & Open-Vocabulary (`yolo.md`)**:
   - Filename: [references/yolo.md](references/yolo.md)
   - Read when: You need performance benchmarks, dual-head details, or usage of YOLOE-26 text-prompting and visual-prompting features.
2. **Python API Guide (`pythonyolo.md`)**:
   - Filename: [references/pythonyolo.md](references/pythonyolo.md)
   - Read when: You need reference code templates for model training, validation, prediction, model exporting, object tracking, benchmarking, and custom training modules.
3. **Hyperparameter & Settings Reference (`configuration.md`)**:
   - Filename: [references/configuration.md](references/configuration.md)
   - Read when: You need a complete list of training, validation, prediction, and export arguments (like `lr0`, `imgsz`, `batch`, `augment`, `end2end`, etc.) and their defaults.
4. **Data Utilities & Dataset Conversions (`utilities.md`)**:
   - Filename: [references/utilities.md](references/utilities.md)
   - Read when: You need to perform auto-labeling (SAM + YOLO), visualize annotations, or convert dataset formats (COCO-to-YOLO, Segmentation masks to YOLO segment format, etc.).
5. **Advanced Trainer Customization (`advanced.md`)**:
   - Filename: [references/advanced.md](references/advanced.md)
   - Read when: You need to customize core components, subclass `BaseTrainer` or `DetectionTrainer`, override model loaders/dataloaders, or freeze backbone layers.
6. **Callbacks & Hook Points (`callback.md`)**:
   - Filename: [references/callback.md](references/callback.md)
   - Read when: You need the full list of event hooks (like `on_train_epoch_end`, `on_model_save`, `on_predict_batch_end`) and examples of how to hook metrics.
