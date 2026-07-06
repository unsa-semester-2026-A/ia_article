# Bibliography PDFs Manifest

Initial bibliography package for the Materials and Methods section of the SMART Challenge 2026 article.

Selection criteria used in this pass:

- Year 2021 or newer, using the journal publication year when available.
- Real downloadable PDF verified locally with `file` and `pdfinfo`.
- Useful for Theoretical Background, Tools and Technologies, or Dataset.
- Journal venue is Q1/Q2 candidate by reputation/ranking category; verify final quartile in the team's required source, such as JCR or SCImago, before final submission.
- PDFs may come from arXiv when the work has a verified journal publication and DOI.

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

## Rejected In This Pass

| Candidate | Reason |
|---|---|
| Object detection using YOLO: challenges, architectural successors, datasets and applications | The Springer link downloaded as HTML in this environment, not as a valid PDF. Do not use it unless a real PDF is obtained later. |

## Still Needed

- Official SMART Challenge/Kaggle citation or rules/source material.
- Dataset statistics after the local dataset finishes downloading.
- Additional 2021+ Q1/Q2 PDFs for traffic video analysis, vehicle detection in urban/UAV scenes, YOLO/OBB tooling, and rotated IoU/AP metrics.
