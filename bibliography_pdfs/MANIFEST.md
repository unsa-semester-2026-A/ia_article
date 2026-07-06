# Bibliography PDFs Manifest

Initial bibliography package for the Materials and Methods section of the SMART Challenge 2026 article.

Selection criteria used in this pass:

- Year 2021 or newer, using the journal publication year when available.
- Real downloadable PDF verified locally with `file` and `pdfinfo`.
- Useful for Theoretical Background, Tools and Technologies, or Dataset.
- Journal venue is Q1/Q2 candidate by reputation/ranking category; verify final quartile in the team's required source, such as JCR or SCImago, before final submission.
- PDFs may come from arXiv when the work has a verified journal publication and DOI.
- A second pass added top-tier conference papers only when they reduce overdependence on broad surveys and support a concrete methodological claim; these are marked as conference sources, not as Q1/Q2 journal replacements.

## Downloaded PDFs

| File | BibTeX key | Paper | Venue | Year | Methodology use |
|---|---|---|---:|---:|---|
| `2021_ding_dota_large-scale-benchmark.pdf` | `Ding2022DOTA` | Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges | IEEE TPAMI | 2022 | Dataset/benchmark context for aerial object detection, OBB annotations, scale variation, orientation, class imbalance, and evaluation motivation. |
| `2021_zand_oriented-bounding-boxes-small-rotated-objects.pdf` | `Zand2022OBB` | Oriented Bounding Boxes for Small and Freely Rotated Objects | IEEE TGRS | 2022 | Theoretical background for OBB, small rotated objects, remote-sensing detection, and why horizontal boxes are insufficient. |
| `2022_han_align-deep-features-oriented-object-detection.pdf` | `Han2022S2ANet` | Align Deep Features for Oriented Object Detection | IEEE TGRS | 2022 | OBB detector design, feature alignment, oriented detection modules, DOTA/HRSC2016 evaluation context. |
| `2022_wen_scale-invariant-mahalanobis-rotated-detection.pdf` | `Wen2022MahalanobisRotatedDetection` | Rotated Object Detection via Scale-Invariant Mahalanobis Distance in Aerial Images | IEEE GRSL | 2022 | Theoretical background for rotated-box regression losses, scale sensitivity, and alternatives to direct angle/box regression. |
| `2021_weber_artificial-images-aerial-vehicle-detection.pdf` | `Weber2021ArtificialImagesVehicleDetection` | Artificial and Beneficial: Exploiting Artificial Images for Aerial Vehicle Detection | ISPRS JPRS | 2021 | Dataset/tools background for synthetic data and data augmentation in aerial vehicle detection; useful for discussing augmentation cautiously. |
| `2023_zou_object-detection-in-20-years-survey.pdf` | `Zou2023ObjectDetectionSurvey` | Object Detection in 20 Years: A Survey | Proceedings of the IEEE | 2023 | General theoretical background for object detection, detector families, evaluation, and development from traditional methods to deep learning. |
| `2023_cheng_large-scale-small-object-detection-survey.pdf` | `Cheng2023SmallObjectSurvey` | Towards Large-Scale Small Object Detection: Survey and Benchmarks | IEEE TPAMI | 2023 | Theoretical background for small-object detection challenges, scale variation, benchmark construction, and why small vehicles are difficult. |
| `2025_huang_task-wise-sampling-convolutions-aood.pdf` | `Huang2025TSConv` | Task-Wise Sampling Convolutions for Arbitrary-Oriented Object Detection in Aerial Images | IEEE TNNLS | 2025 | Advanced OBB detection challenges: task inconsistency, localization/classification alignment, oriented feature sampling. |
| `2025_jamali_context-object-detection-review.pdf` | `Jamali2025ContextObjectDetection` | Context in Object Detection: A Systematic Literature Review | Artificial Intelligence Review | 2025 | Theoretical background for contextual information in detection, useful for discussing traffic-scene context, occlusion, and surrounding cues. |
| `2025_zhou_traffic-surveillance-vision-survey.pdf` | `Zhou2025TrafficSurveillanceSurvey` | Vision Technologies with Applications in Traffic Surveillance Systems: A Holistic Survey | ACM Computing Surveys | 2025 | Domain background for traffic surveillance systems, computer vision pipelines, traffic scene understanding, and intelligent transportation. |
| `2025_nikouei_small-object-detection-comprehensive-survey.pdf` | `Nikouei2025SmallObjectSurvey` | Small Object Detection: A Comprehensive Survey on Challenges, Techniques and Real-World Applications | Intelligent Systems with Applications | 2025 | Additional support for small-object detection challenges and real-world applications, including scale, resolution, and dense scenes. |
| `2025_wang_oriented-object-detection-survey.pdf` | `Wang2025OBBSurvey` | Oriented object detection in optical remote sensing images using deep learning: a survey | Artificial Intelligence Review | 2025 | Broad survey for theoretical background: HBB-to-OBB evolution, OBB regression, datasets, evaluation protocols, and open challenges. |
| `2021_xie_oriented-rcnn-object-detection.pdf` | `Xie2021OrientedRCNN` | Oriented R-CNN for Object Detection | ICCV | 2021 | Specific two-stage OBB detector reference; useful to avoid citing only OBB surveys when discussing oriented proposals and efficient oriented detection. |
| `2021_han_redet-rotation-equivariant-detector.pdf` | `Han2021ReDet` | ReDet: A Rotation-equivariant Detector for Aerial Object Detection | CVPR | 2021 | Specific OBB detector reference for rotation-equivariant features and orientation modeling in aerial object detection. |
| `2021_yang_kld-rotated-object-detection.pdf` | `Yang2021KLD` | Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence | NeurIPS | 2021 | Specific rotated-box regression/loss reference for high-precision localization and angle-sensitive evaluation. |
| `2022_akyon_sahi-small-object-detection.pdf` | `Akyon2022SAHI` | Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection | ICIP | 2022 | Specific small-object detection technique for slicing high-resolution imagery and improving detection of far/small objects. |
| `2024_azfar_complex-traffic-environments-review.pdf` | `Azfar2024ComplexTrafficReview` | Deep Learning-Based Computer Vision Methods for Complex Traffic Environments Perception: A Review | Data Science for Transportation | 2024 | Viability/scalability support for real-world traffic perception: labeling needs, high data volume, real-time constraints, embedded hardware, lighting, occlusion, heterogeneous traffic, and camera viewpoints. |
| `2021_koga_adapting-vehicle-detector-domain.pdf` | `Koga2021VehicleDomainAdaptation` | Adapting Vehicle Detector to Target Domain by Adversarial Prediction Alignment | IGARSS | 2021 | Domain adaptation support for vehicle detectors when target-domain conditions differ from the training domain. |
| `2021_vora_generalization-multiview-detection.pdf` | `Vora2023GeneralizationMultiview` | Bringing Generalization to Deep Multi-View Pedestrian Detection | WACVW | 2023 | Generalization support across camera positions, camera counts, and new scenes; useful for discussing scalability across viewpoints and deployment locations. |

## Rejected In This Pass

| Candidate | Reason |
|---|---|
| Object detection using YOLO: challenges, architectural successors, datasets and applications | The Springer link downloaded as HTML in this environment, not as a valid PDF. Do not use it unless a real PDF is obtained later. |
| Adaptive Slicing-Aided Hyper Inference for Small Object Detection in High-Resolution Remote Sensing Images | MDPI blocked direct PDF download in this environment and returned an access-denied HTML file; do not use unless a real PDF is obtained later. |
| Deep Learning-Based Computer Vision Methods for Complex Traffic Environments Perception: A Review, Springer direct PDF | Springer direct PDF URL returned HTML in this environment; the arXiv PDF was downloaded instead while the BibTeX records the formal Springer journal DOI. |

## Still Needed

- Official SMART Challenge/Kaggle citation or rules/source material.
- Dataset statistics after the local dataset finishes downloading.
- Additional 2021+ Q1/Q2 PDFs for traffic video analysis, vehicle detection in urban/UAV scenes, YOLO/OBB tooling, and rotated IoU/AP metrics.
