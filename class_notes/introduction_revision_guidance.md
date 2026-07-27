# Introduction Revision Guidance

Do not edit `latex_report/content/introduction.tex` until the team approves a rewrite. The current introduction still needs revision to align with the updated SAM copy-paste methodology and the instructor observations.

Recommended correction:

1. Start from the general problem of camera-based vehicle surveillance in Lima, not from the MTC competition or the dataset.
2. Remove early dataset-specific details from the introduction. Dataset size, class counts, file structure, and annotation format belong in the Dataset section.
3. Reframe the research gap around generalization under real urban camera variability: viewpoint changes, different CCTV placements, illumination shifts, scale differences, occlusions, traffic density, and domain shift.
4. Replace claims about YOLO11 vs. YOLO26 ablation if those experiments are not present in the current final pipeline. The implementation inspected here uses YOLO26s-OBB for the final F1 training family.
5. Add a conceptual bridge from class imbalance and small oriented vehicles to segmentation-assisted augmentation. Explain that the proposed solution increases minority-class training evidence by extracting real vehicle masks and inserting them into train-only safe slots.
6. State clearly that IC-Light was explored preliminarily but is not part of the final production augmentation release.
7. End with one continuous contributions paragraph, not a numbered list. The contribution should mention dataset audit, OBB conversion, SAM-based copy-paste release, validation-preserving design, Macro AP-rIoU evaluation code, and final evaluation packaging.
8. Keep the final introduction at approximately five to eight developed paragraphs. Avoid one-sentence paragraphs, promotional language, and unverified performance claims. The archived final metrics currently report zero true positives and zero Macro AP-rIoU for all inspected conditions, so the introduction must not suggest that the augmentation improved detection performance.

Specific outdated claims that should be removed or rewritten:

- The current introduction says the work compares YOLO11-OBB and YOLO26-OBB. The inspected final training family is YOLO26s-OBB, and the archived evaluation packages report Base 0, Base 1, Base 2, Improvement A, Improvement B, and Improvement C.
- The current introduction says class-weighted focal loss is introduced. The inspected final F1 runner uses the documented YOLO26s-OBB training conditions; focal loss should not be presented as a completed contribution unless a corresponding implemented run and result artifact are verified.
- The current introduction promises quantitative results and an edge-hardware benchmark. The final artifact store provides executable evaluation packages, but the metric summaries report zero true positives and zero Macro AP-rIoU; therefore, any performance-improvement claim should be removed.
- The current contribution paragraph should be reframed around the actual repository evidence: dataset audit, OBB conversion, SAM copy-paste release, train-only validation isolation, final evaluation packaging, and the unresolved metric-alignment audit.

Suggested final flow:

1. Importance of computer-vision traffic surveillance for urban mobility.
2. Detection problems in real Lima-like camera scenes.
3. Generalization across cameras, viewpoints, illumination, scale, occlusion, and density.
4. Limits of training only on observed frames when classes are long-tailed.
5. Motivation for segmentation-assisted data augmentation.
6. Conceptual description of the SAM copy-paste solution.
7. Explanation of train-only validation-preserving design and OBB label preservation.
8. Objective, scope, and contributions in one academic paragraph.
