Con el reto del MTC (SMART Challenge 2026) como filtro, de las ~40 referencias del CSV yo priorizaría un grupo pequeño y bien enfocado, en vez de leer todo el listado. Te los agrupo según qué parte del pipeline te resuelven:

**1. Para entender el panorama general de OBB (lectura base, antes que nada)**
- *"Oriented object detection in optical remote sensing images using deep learning: a survey"* (Wang et al., 2023) — es la revisión más reciente y completa; te da el mapa mental de todas las familias de métodos (anchor-based, anchor-free, representaciones angulares) y sus problemas típicos. Ideal como punto de partida antes de elegir arquitectura (ítem 1 del prompt).
- *"Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges"* (DOTA, Ding et al., 2021) — es el dataset/benchmark de referencia del campo; útil para entender baselines y cómo se comparan arquitecturas.

**2. Los más directamente análogos a tu problema (vehículos + desbalance de clases)**
- *"Improved Faster RCNN Based on Feature Amplification and Oversampling Data Augmentation for Oriented Vehicle Detection"* (Mo & Yan, 2020) — trabaja sobre VEDAI, que tiene 9 categorías de vehículos igual que tu reto, y ataca justo el desbalance de clases con oversampling + stitching. Muy aplicable al ítem 4.
- *"Research on Vehicle Detection in Infrared Aerial Images in Complex Urban and Road Backgrounds"* (Yu et al., 2024) — construyen un dataset balanceado ("BalancedVehicle") sobre YOLOv5-obb para resolver exactamente el problema de distribución desigual entre tipos de vehículo. Buen complemento al anterior.
- *"UAV Video Vehicle Detection: Benchmark and Baseline"* (Xiao et al., 2025) — este es probablemente el más parecido conceptualmente a tu competencia: video UAV, OBB, categorías de vehículo, y explícitamente separan detección de clasificación de categoría. Útil para el ítem 3 (aprovechamiento temporal del clip).

**3. Para la métrica y la función de pérdida (rotated IoU)**
- *"Building a Bridge of Bounding Box Regression Between Oriented and Horizontal Object Detection"* (Qian et al., 2023) — proponen la pérdida RIoU (rotated IoU) directamente, que es justo la base conceptual de tu métrica oficial Macro AP-rIoU. Muy relevante para el ítem 5 (implementar tu evaluador local).

**4. Para elegir arquitectura concreta (ítem 1)**
- *"FCOSR: A Simple Anchor-free Rotated Detector for Aerial Object Detection"* — buen candidato si buscas algo simple, rápido y desplegable (mencionan hasta Jetson), relevante si el reto valora también velocidad de inferencia.
- *"A Modified YOLOv8 Detection Network for UAV Aerial Image Recognition"* (Li et al., 2023) — específicamente aborda drones + objetos pequeños + WiseIoU loss, cercano a lo que verías en frames de intersecciones urbanas.
- *"Vehicle Detection and Classification via YOLOv8 and Deep Belief Network"* (Al Mudawi et al., 2023) — pipeline vehicular completo con YOLOv8, útil como referencia de arquitectura extremo a extremo.

**Yo dejaría de lado** (poco o nada aplicables a tu caso): el paper de LiDAR para drones, el de paneles de arroz (rice panicle), HDDet (heading direction genérico), y varios papers de SAR/naval (ships) que están más lejos del dominio vehicular urbano.

Si quieres, puedo armarte una tabla comparativa (arquitectura, dataset de prueba, mAP, si maneja desbalance de clases) para que decidas más rápido cuál usar como base de tu modelo.
