# Prototipado y Cálculo de Homografías al Vuelo (On-the-Fly Ego-Motion Compensation)

Este notebook en formato Markdown está diseñado para ser copiado y pegado paso a paso en **Google Colab**. Guía la implementación y verificación del cálculo de homografías **al vuelo (on-the-fly)** utilizando las predicciones de un modelo detector OBB ligero para generar las máscaras de supresión de vehículos.

---

### Paso 1: Configuración del Entorno e Importaciones
Instalamos e importamos la librería `ultralytics` para poder cargar y ejecutar la inferencia de YOLO OBB. También montamos Google Drive para persistir los resultados.

```python
# Install Ultralytics library
!pip install -q ultralytics

# Import standard computer vision and data libraries
import os
import cv2
import json
import time
import zipfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Try to mount Google Drive if running in Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive mounted successfully.")
except ImportError:
    print("Not running in Google Colab. Skipping Drive mount.")
```

---

### Paso 2: Resolución del Split de Validación y Extracción Selectiva
Cargamos `split_metadata.csv` de Drive para saber qué videos corresponden a la validación. Luego, extraemos únicamente las imágenes de **4 clips de validación** de `train.zip` para realizar este experimento piloto sin saturar el almacenamiento de la máquina virtual de Colab.

```python
# Configure dataset directories and output paths
VAL_IMAGES_DIR = "/content/dataset/val/images"
METADATA_PATH = "/content/drive/MyDrive/ia_article/01_processed/split_metadata.csv"
ZIP_PATH = "/content/drive/MyDrive/ia_article/00_raw/train.zip"
OUTPUT_DIR = "/content/drive/MyDrive/ia_article/05_evaluations"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "validation_homographies.json")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)

# 1. Read metadata to find validation clips
if os.path.exists(METADATA_PATH):
    metadata_df = pd.read_csv(METADATA_PATH)
    val_clips = metadata_df[metadata_df["split"] == "val"]["clip_id"].unique().tolist()
    print(f"✓ Found {len(val_clips)} total validation clips in split_metadata.csv.")
else:
    raise FileNotFoundError(f"Metadata file not found at: {METADATA_PATH}")

# 2. Select 4 prototype clips
prototype_clips = sorted(val_clips)[:4]
print(f"Prototype validation clips selected: {prototype_clips}")

# 3. Open zip container and extract only the selected clips' frames
print("Extracting frames from train.zip... (This may take a minute)")
start_extract = time.time()
extracted_count = 0

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    zip_files = z.namelist()
    for name in zip_files:
        # Check if the file is a JPG inside the 'train/' folder of the zip
        if name.endswith(".jpg") and name.startswith("train/"):
            filename = os.path.basename(name)
            # Check if this frame belongs to one of our prototype clips
            for clip in prototype_clips:
                if filename.startswith(clip + "_"):
                    with z.open(name) as src, open(os.path.join(VAL_IMAGES_DIR, filename), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted_count += 1
                    break

print(f"✓ Extracted {extracted_count} frames in {time.time() - start_extract:.2f} seconds.")
```

---

### Paso 3: Funciones de Enmascaramiento Dinámico y Estimación de Homografía
Definimos los módulos para calcular las homografías al vuelo:
* **Enmascaramiento Dinámico:** Recibe las predicciones OBB del modelo (en formato `xywhr` de Ultralytics, donde la rotación viene en **radianes**) y las dibuja en negro (0) sobre la máscara binaria. Esto excluye dinámicamente los autos detectados.
* **Cálculo con Fallback:** Calcula la homografía inter-frame y cae de forma segura en `np.eye(3)` si hay baja textura o fallos de RANSAC.

```python
def create_exclusion_mask_from_predictions(
    img_shape: tuple, 
    obb_predictions
) -> np.ndarray:
    """Generates a binary mask where predicted vehicle boxes are 0 (black) and background is 255 (white).

    Args:
        img_shape: Tuple representing (height, width) of the image.
        obb_predictions: Numpy array of shape (N, 5) representing [cx, cy, w, h, angle_rad] from model.
    """
    # Start with a solid white mask (all pixels active)
    mask = np.ones(img_shape[:2], dtype=np.uint8) * 255
    
    if obb_predictions is None or len(obb_predictions) == 0:
        return mask
        
    for box in obb_predictions:
        cx, cy, w, h, angle_rad = box
        
        # Calculate corners from center, width, height, and angle in radians
        cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
        
        # Local offsets for the 4 corners relative to the center
        dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
        dy = np.array([-h / 2, -h / 2, h / 2, h / 2])
        
        # Rotate and translate
        x_rot = dx * cos_t - dy * sin_t + cx
        y_rot = dx * sin_t + dy * cos_t + cy
        
        # Build polygon array in integer pixel coordinates
        corners = np.stack((x_rot, y_rot), axis=1).astype(np.int32)
        
        # Paint the vehicle polygon black (0) on the mask
        cv2.fillPoly(mask, [corners], 0)
        
    return mask


def estimate_interframe_homography(
    prev_gray: np.ndarray, 
    curr_gray: np.ndarray,
    prev_mask: np.ndarray = None
) -> np.ndarray:
    """Estimates the 3x3 homography matrix. Prevents vehicle feature pollution using a mask.

    If matching fails, falls back to the 3x3 Identity matrix.
    """
    orb = cv2.ORB_create(nfeatures=1500)
    
    # Extract features applying the exclusion mask on the previous frame
    kp1, des1 = orb.detectAndCompute(prev_gray, mask=prev_mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    
    # Fallback if keypoints are insufficient
    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return np.eye(3)
        
    # Match using BF Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 10:
        return np.eye(3)
        
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimate homography with RANSAC
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None or np.sum(status) < 10:
        return np.eye(3)
        
    return H
```

---

### Paso 4: Carga del Modelo Ligero e Inferencia en Bucle al Vuelo
Cargamos el modelo oficial ligero `yolov8s-obb.pt` (se descargará de forma automática). Iteramos sobre los frames del primer clip, ejecutamos la predicción en cada frame anterior para generar dinámicamente la máscara de supresión de autos y calculamos las homografías inter-frame en tiempo real. 

Para verificar la precisión visual sin ralentizar el bucle, generaremos una imagen superpuesta al 50% de opacidad únicamente para la primera transición (`i == 0`) de cada clip.

```python
# 1. Load lightweight OBB model
print("Loading lightweight OBB model...")
model = YOLO("yolov8s-obb.pt")

# Group local images into clips
def group_local_images(images_dir: str, target_clips: list) -> dict:
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    clips = {}
    for filename in sorted(image_files):
        parts = filename.split('_')
        clip_id = "_".join(parts[:-1])
        if clip_id in target_clips:
            clips.setdefault(clip_id, []).append(filename)
            
    for c in clips:
        clips[c].sort()
    return clips

local_clips = group_local_images(VAL_IMAGES_DIR, prototype_clips)

all_results = {}

for clip_id, frame_list in local_clips.items():
    print(f"\nProcessing clip '{clip_id}' ({len(frame_list)} frames) on-the-fly...")
    clip_homographies = {}
    
    # Load first frame
    first_frame_path = os.path.join(VAL_IMAGES_DIR, frame_list[0])
    prev_img = cv2.imread(first_frame_path)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    
    start_time = time.time()
    H = np.eye(3)  # Initial state
    
    for i in range(len(frame_list) - 1):
        prev_name = frame_list[i]
        curr_name = frame_list[i + 1]
        
        # 2. Predict on previous frame to locate vehicles
        results = model.predict(prev_img, conf=0.05, verbose=False)
        
        # Extract prediction boxes in xywhr format
        obb_boxes = results[0].obb
        if obb_boxes is not None and len(obb_boxes) > 0:
            # Convert tensor to numpy array: [cx, cy, w, h, angle_rad]
            pred_boxes_np = obb_boxes.xywhr.cpu().numpy()
        else:
            pred_boxes_np = None
            
        # 3. Generate exclusion mask dynamically from predictions
        prev_mask = create_exclusion_mask_from_predictions(prev_gray.shape, pred_boxes_np)
        
        # Load current frame
        curr_img_path = os.path.join(VAL_IMAGES_DIR, curr_name)
        curr_img = cv2.imread(curr_img_path)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        
        # 4. Compute homography matrix using prediction mask
        H = estimate_interframe_homography(prev_gray, curr_gray, prev_mask)
        
        # Save transformation
        transition_key = f"{prev_name}->{curr_name}"
        clip_homographies[transition_key] = H.tolist()
        
        # ─── VALIDACIÓN VISUAL DE ALINEACIÓN (SOLO LA PRIMERA TRANSICIÓN DEL CLIP) ───
        if i == 0:
            # Project the true previous frame onto the plane of the current frame using H
            warped_prev = cv2.warpPerspective(prev_img, H, (curr_img.shape[1], curr_img.shape[0]))
            # Combine current image and aligned previous image (50% opacity each)
            overlay = cv2.addWeighted(curr_img, 0.5, warped_prev, 0.5, 0)
            # Save the overlay validation image locally
            validation_img_path = f"/content/validation_alignment_{clip_id}.jpg"
            cv2.imwrite(validation_img_path, overlay)
            print(f"  ✓ First transition visual alignment saved to: {validation_img_path}")
            
        # Advance state
        prev_img = curr_img
        prev_gray = curr_gray
        
    duration = time.time() - start_time
    print(f"  Clip completed in {duration:.2f}s ({duration/len(frame_list)*1000:.1f} ms/frame)")
    all_results[clip_id] = clip_homographies
```

---

### Paso 5: Serialización en Google Drive e Inspección de Matrices
Para terminar, guardamos el JSON estructurado final de las homografías calculadas al vuelo en Google Drive y mostramos una matriz de transformación de ejemplo.

```python
# 1. Save prototype outputs to Google Drive
with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Dynamic validation homographies JSON successfully saved to Google Drive: {OUTPUT_JSON_PATH}")

# 2. Inspect a sample matrix from the output
sample_clip = list(all_results.keys())[0]
sample_transition = list(all_results[sample_clip].keys())[0]
print(f"\nSample Homography Matrix for '{sample_transition}':")
print(json.dumps(all_results[sample_clip][sample_transition], indent=2))
```
