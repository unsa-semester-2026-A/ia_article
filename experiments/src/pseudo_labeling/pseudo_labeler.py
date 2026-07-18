#!/usr/bin/env python3
"""Production Pseudo-Labeling and Static Vehicle Detection Module.

This script runs zero-shot YOLO-OBB inference using pre-trained DOTA weights,
applies ORB+RANSAC homography motion compensation, tracks vehicle centroids
temporally, and cross-references detections with the ground truth to isolate
static, unannotated parked vehicles. It features clip-level checkpointing
with atomic writes, direct API uploads, and automated Google Colab runtime shutdown.
"""

import gc
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO


# Dynamic mapping helper for vehicle indices in DOTA
def get_vehicle_class_ids(model: YOLO) -> List[int]:
    """Finds DOTA model class indices corresponding to vehicles."""
    vehicle_class_ids = []
    for idx, name in model.names.items():
        clean_name = name.lower().replace("-", " ").replace("_", " ")
        if any(
            keyword in clean_name
            for keyword in [
                "small vehicle",
                "large vehicle",
                "car",
                "bus",
                "truck",
                "vehicle",
            ]
        ):
            vehicle_class_ids.append(int(idx))
    return vehicle_class_ids


# Homography calculation
def estimate_homography(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_detections: List[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """Estimates the 3x3 homography matrix between prev and curr frames, excluding dynamic foreground.

    Args:
        prev_gray: Grayscale previous frame.
        curr_gray: Grayscale current frame.
        prev_detections: Bounding box polygons to mask from feature tracking.

    Returns:
        3x3 Homography matrix, and boolean success flag.
    """
    orb = cv2.ORB_create(nfeatures=2500)
    bg_mask = np.ones_like(prev_gray, dtype=np.uint8) * 255

    if prev_detections is not None:
        for det in prev_detections:
            poly = np.array(det, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(bg_mask, [poly], 0)

    kp1, des1 = orb.detectAndCompute(prev_gray, mask=bg_mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, mask=None)

    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        return np.eye(3), False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        return np.eye(3), False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None or np.sum(status) < 10:
        return np.eye(3), False

    return H, True


def project_centroid(cx: float, cy: float, H: np.ndarray) -> Tuple[float, float]:
    """Projects a point (cx, cy) using homography matrix H."""
    pt = np.array([cx, cy, 1.0]).reshape(3, 1)
    projected = np.dot(H, pt)
    w = projected[2, 0]
    if abs(w) > 1e-6:
        return projected[0, 0] / w, projected[1, 0] / w
    return cx, cy


# Simple Centroid Tracker
class CentroidTracker:
    """Tracks bounding box centroids across sequential frames using motion projection."""

    def __init__(self, max_distance: float = 30.0):
        """Initializes the CentroidTracker with a maximum proximity distance."""
        self.next_object_id = 0
        self.objects: Dict[int, Tuple[float, float]] = {}
        self.history: Dict[int, List[Tuple]] = {}
        self.max_distance = max_distance

    def update(
        self,
        frame_idx: int,
        detections: List[Tuple[float, float, float, float, float, float]],
        H: np.ndarray,
    ):
        """Updates trackers using newly detected objects in current frame.

        Args:
            frame_idx: True frame number.
            detections: List of (cx, cy, w, h, angle, conf) boxes.
            H: Homography matrix mapping previous coordinates to current frame coordinates.
        """
        projected_objects = {}
        for obj_id, centroid in self.objects.items():
            pcx, pcy = project_centroid(centroid[0], centroid[1], H)
            projected_objects[obj_id] = (pcx, pcy)

        new_objects = {}
        assigned_det_indices = set()

        if len(projected_objects) > 0 and len(detections) > 0:
            object_ids = list(projected_objects.keys())
            object_points = np.array(list(projected_objects.values()))
            det_points = np.array([[d[0], d[1]] for d in detections])

            for i, obj_id in enumerate(object_ids):
                dists = np.linalg.norm(det_points - object_points[i], axis=1)
                min_idx = np.argmin(dists)

                if (
                    dists[min_idx] < self.max_distance
                    and min_idx not in assigned_det_indices
                ):
                    det = detections[min_idx]
                    new_objects[obj_id] = (det[0], det[1])
                    self.history[obj_id].append((frame_idx, *det))
                    assigned_det_indices.add(min_idx)

        for idx, det in enumerate(detections):
            if idx not in assigned_det_indices:
                new_id = self.next_object_id
                self.next_object_id += 1
                new_objects[new_id] = (det[0], det[1])
                self.history[new_id] = [(frame_idx, *det)]

        self.objects = new_objects


# Ground Truth checks and filtering
def check_gt_matching(
    tracked_cx: float,
    tracked_cy: float,
    gt_df: pd.DataFrame,
    frame_idx: int,
    dist_threshold: float = 30.0,
) -> bool:
    """Checks if a tracked centroid matches an annotated moving vehicle in the ground truth."""
    frame_gt = gt_df[gt_df["frame_idx"] == frame_idx]
    if frame_gt.empty:
        return False
    gt_coords = frame_gt[["cx", "cy"]].values
    dists = np.linalg.norm(gt_coords - np.array([tracked_cx, tracked_cy]), axis=1)
    return bool(np.any(dists < dist_threshold))


def filter_static_vehicles(
    tracker_history: Dict[int, List[Tuple]],
    gt_df: pd.DataFrame,
    min_frames: int = 10,
    motion_threshold: float = 8.0,
) -> Dict[str, List[Dict]]:
    """Filters tracked trajectories to isolate static vehicles that do not match the GT."""
    static_detections = {}
    for obj_id, traj in tracker_history.items():
        if len(traj) < min_frames:
            continue

        coords = np.array([[pt[1], pt[2]] for pt in traj])
        dx = np.max(coords[:, 0]) - np.min(coords[:, 0])
        dy = np.max(coords[:, 1]) - np.min(coords[:, 1])
        drift = math.sqrt(dx**2 + dy**2)

        if drift < motion_threshold:
            has_gt_match = False
            for step in traj:
                frame_idx, cx, cy = step[0], step[1], step[2]
                if check_gt_matching(cx, cy, gt_df, frame_idx):
                    has_gt_match = True
                    break

            if not has_gt_match:
                for step in traj:
                    frame_idx, cx, cy, w, h, angle, conf = step
                    frame_name = f"frame_{frame_idx:04d}"
                    if frame_name not in static_detections:
                        static_detections[frame_name] = []
                    static_detections[frame_name].append(
                        {
                            "cx": float(cx),
                            "cy": float(cy),
                            "w": float(w),
                            "h": float(h),
                            "angle": float(angle),
                            "conf": float(conf),
                        }
                    )
    return static_detections


class PseudoLabeler:
    """Orchestrates the temporal pseudo-labeling pipeline for static vehicle detection."""

    def __init__(
        self,
        csv_path: str = "/content/drive/MyDrive/ia_article/00_raw/train.csv",
        zip_path: str = "/content/drive/MyDrive/ia_article/00_raw/train.zip",
        metadata_path: str = "/content/drive/MyDrive/ia_article/01_processed/split_metadata.csv",
        output_dir: str = "/content/tmp_pseudo",
        model_name: str = "yolo26m-obb.pt",
        conf_threshold: float = 0.25,
        min_frames: int = 10,
        motion_threshold: float = 8.0,
        batch_size: int = 16,
        token_path: str = "/content/drive/MyDrive/ia_article/token/token.json",
        folder_id: str = "1J5ogC3q6jyYlk3wuYyxpYZHslUg6eGtN",
        checkpoints_folder_id: str = "1anPtHNwHYgcq4BImhbJ_xzouiJ-Sh035",
    ):
        """Initializes the PseudoLabeler with required paths and execution options."""
        self.csv_path = csv_path
        self.zip_path = zip_path
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.min_frames = min_frames
        self.motion_threshold = motion_threshold
        self.batch_size = batch_size
        self.token_path = token_path
        self.folder_id = folder_id
        self.checkpoints_folder_id = checkpoints_folder_id

    def _get_drive_service(self):
        """Helper to build Drive API service using team authorized credentials JSON."""
        if not os.path.exists(self.token_path):
            return None
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            creds = Credentials.from_authorized_user_file(
                self.token_path, ["https://www.googleapis.com/auth/drive.file"]
            )
            return build("drive", "v3", credentials=creds)
        except Exception as e:
            print(f"Error initializing Google Drive API: {e}", file=sys.stderr)
            return None

    def _get_processed_clips_from_drive(self, service) -> Set[str]:
        """Queries checkpoints folder in Drive with pagination to find all processed clips."""
        if not service:
            return set()
        processed = set()
        page_token = None
        try:
            while True:
                query = f"'{self.checkpoints_folder_id}' in parents and trashed = false"
                results = (
                    service.files()
                    .list(
                        q=query,
                        pageSize=1000,
                        fields="nextPageToken, files(id, name)",
                        pageToken=page_token,
                    )
                    .execute()
                )

                for f in results.get("files", []):
                    if f["name"].endswith(".json"):
                        processed.add(f["name"].split(".")[0])

                page_token = results.get("nextPageToken", None)
                if page_token is None:
                    break
            return processed
        except Exception as e:
            print(f"Error listing checkpoints from Google Drive: {e}", file=sys.stderr)
            return set()

    def _upload_file_to_drive(
        self,
        service,
        local_path: str,
        parent_folder_id: str,
        mime_type: str = "application/json",
    ):
        """Uploads a local file to a specific Drive folder ID using resumable upload API."""
        if not service:
            return None
        try:
            from googleapiclient.http import MediaIoBaseUpload

            filename = os.path.basename(local_path)
            with open(local_path, "rb") as f:
                file_data = f.read()
            flujo_archivo = io.BytesIO(file_data)
            metadatos = {
                "name": filename,
                "mimeType": mime_type,
                "parents": [parent_folder_id],
            }
            media = MediaIoBaseUpload(flujo_archivo, mimetype=mime_type, resumable=True)
            archivo = (
                service.files()
                .create(body=metadatos, media_body=media, fields="id")
                .execute()
            )
            return archivo.get("id")
        except Exception as e:
            print(
                f"Error uploading file {local_path} to Google Drive: {e}",
                file=sys.stderr,
            )
            return None

    def _download_missing_checkpoints(self, service, local_checkpoint_dir: str):
        """Downloads missing JSON checkpoint files from Drive to local directory with pagination."""
        if not service:
            return
        page_token = None
        try:
            while True:
                query = f"'{self.checkpoints_folder_id}' in parents and trashed = false"
                results = (
                    service.files()
                    .list(
                        q=query,
                        pageSize=1000,
                        fields="nextPageToken, files(id, name)",
                        pageToken=page_token,
                    )
                    .execute()
                )

                files = results.get("files", [])
                for f in files:
                    filename = f["name"]
                    if not filename.endswith(".json"):
                        continue
                    local_path = os.path.join(local_checkpoint_dir, filename)
                    if os.path.exists(local_path):
                        continue  # Already downloaded

                    file_id = f["id"]
                    request = service.files().get_media(fileId=file_id)
                    content = request.execute()
                    with open(local_path, "wb") as fh:
                        fh.write(content)

                page_token = results.get("nextPageToken", None)
                if page_token is None:
                    break
        except Exception as e:
            print(
                f"Error downloading checkpoints from Google Drive: {e}", file=sys.stderr
            )

    def run(self) -> str:
        """Runs the complete pseudo-labeling pipeline over the training split clips."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            service = self._get_drive_service()
            if service:
                print("✓ Google Drive API service initialized.")
                drive_processed_clips = self._get_processed_clips_from_drive(service)
                print(
                    f"✓ Found {len(drive_processed_clips)} processed clips on Google Drive."
                )
            else:
                print(
                    "Warning: Google Drive API service not initialized or token missing. Running in local-only mode."
                )
                drive_processed_clips = set()

            print("Loading pre-trained YOLO OBB model...")
            model = YOLO(self.model_name)
            vehicle_class_ids = get_vehicle_class_ids(model)
            print(
                f"Dynamically mapped vehicle class IDs from {self.model_name}: {vehicle_class_ids}"
            )

            print("Loading annotations dataset...")
            df_raw = pd.read_csv(self.csv_path)
            df_raw["clip_id"] = df_raw["Id"].apply(
                lambda x: "_".join(x.split("_")[:-1])
            )

            # Filter only train split clips if metadata is present
            if os.path.exists(self.metadata_path):
                metadata_df = pd.read_csv(self.metadata_path)
                train_clips = set(
                    metadata_df[metadata_df["split"] == "train"]["clip_id"].unique()
                )
                unique_clips = sorted(
                    [c for c in df_raw["clip_id"].unique() if c in train_clips]
                )
                print(
                    f"✓ split_metadata.csv found. Filtering for {len(unique_clips)} 'train' split clips (val clips skipped)."
                )
            else:
                unique_clips = sorted(df_raw["clip_id"].unique())
                print(
                    f"Warning: split_metadata.csv not found at {self.metadata_path}. Processing all {len(unique_clips)} clips."
                )

            # Filter out parsed GT annotations for fast cross-referencing
            parsed_records = []
            print("Pre-parsing ground truth coordinates for validation check...")
            df_raw_sub = df_raw[df_raw["clip_id"].isin(unique_clips)].copy()
            for _, row in tqdm(df_raw_sub.iterrows(), total=len(df_raw_sub)):
                frame_id = row["Id"]
                target = row["Target"]
                parts = frame_id.split("_")
                clip_id = "_".join(parts[:-1])
                frame_idx = int(parts[-1])

                if target == "none" or pd.isna(target):
                    continue
                for ann in target.split(";"):
                    parts_ann = ann.strip().split(" ")
                    if len(parts_ann) == 6:
                        parsed_records.append(
                            {
                                "Id": frame_id,
                                "clip_id": clip_id,
                                "frame_idx": frame_idx,
                                "cx": float(parts_ann[1]),
                                "cy": float(parts_ann[2]),
                            }
                        )
            gt_df = (
                pd.DataFrame(parsed_records)
                if parsed_records
                else pd.DataFrame(columns=["Id", "clip_id", "frame_idx", "cx", "cy"])
            )

            print(f"Total video clips to process: {len(unique_clips)}")

            # Open zip container once
            with zipfile.ZipFile(self.zip_path, "r") as z:
                zip_files = z.namelist()

                for clip in tqdm(unique_clips, desc="Processing video clips"):
                    checkpoint_path = os.path.join(checkpoint_dir, f"{clip}.json")

                    # Skip if either local or Drive checkpoint exists
                    if os.path.exists(checkpoint_path) or clip in drive_processed_clips:
                        continue

                    clip_frames = sorted(
                        [
                            f
                            for f in zip_files
                            if clip in f and f.endswith((".jpg", ".png"))
                        ]
                    )
                    if not clip_frames:
                        continue

                    tracker = CentroidTracker()
                    prev_gray = None
                    prev_dets_poly = None
                    total_detections = 0
                    homography_failures = 0

                    # Declare lists outside of try to prevent UnboundLocalError in finally block
                    frames_bgr = []
                    frames_gray = []
                    frame_indices = []

                    try:
                        # 1. IO BURST READ: Load all frames into RAM sequentially
                        for frame_zip_path in clip_frames:
                            frame_id_str = os.path.basename(frame_zip_path).split(".")[
                                0
                            ]
                            frame_indices.append(int(frame_id_str.split("_")[-1]))

                            with z.open(frame_zip_path) as file:
                                img_bytes = file.read()
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            frames_bgr.append(img)
                            frames_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

                        # 2. BATCHED GPU INFERENCE: Run model on the batch list
                        results = model.predict(
                            frames_bgr,
                            conf=self.conf_threshold,
                            batch=self.batch_size,
                            verbose=False,
                        )

                        # 3. CPU STATEFUL UPDATE: Sequentially calculate homography and tracker
                        for i, r in enumerate(results):
                            gray = frames_gray[i]
                            frame_idx = frame_indices[i]

                            if i > 0:
                                H, success = estimate_homography(
                                    prev_gray, gray, prev_dets_poly
                                )
                                if not success:
                                    homography_failures += 1
                        else:
                            H = np.eye(3)

                        detections = []
                        current_polys = []

                        if r.obb is not None:
                            for box in r.obb:
                                cls_id = int(box.cls[0].item())
                                if cls_id in vehicle_class_ids:
                                    xywhr = box.xywhr[0].cpu().numpy()
                                    conf = float(box.conf[0].item())
                                    cx, cy, w, h, angle = xywhr

                                    # Convert Radians to Degrees
                                    angle_deg = math.degrees(angle)
                                    detections.append((cx, cy, w, h, angle_deg, conf))
                                    current_polys.append(box.xyxyxyxy[0].cpu().numpy())
                                    total_detections += 1

                        tracker.update(frame_idx, detections, H)
                        prev_gray = gray
                        prev_dets_poly = current_polys

                        # Filter static vehicles
                        clip_gt_df = gt_df[gt_df["clip_id"] == clip]
                        clip_static_map = filter_static_vehicles(
                            tracker.history,
                            clip_gt_df,
                            self.min_frames,
                            self.motion_threshold,
                        )

                        # Structure output keys
                        global_static_map = {}
                        for local_frame_name, boxes in clip_static_map.items():
                            frame_idx_str = local_frame_name.split("_")[-1]
                            frame_id = f"{clip}_{frame_idx_str}"
                            global_static_map[frame_id] = boxes

                        # 4. ATOMIC CHECKPOINT WRITE: Write to .tmp first and replace
                        temp_checkpoint_path = checkpoint_path + ".tmp"
                        with open(temp_checkpoint_path, "w") as f:
                            json.dump(global_static_map, f, indent=2)
                        os.replace(temp_checkpoint_path, checkpoint_path)

                        # 5. PERSISTENCE TO DRIVE: Upload clip checkpoint to Google Drive shared folder
                        if service:
                            self._upload_file_to_drive(
                                service, checkpoint_path, self.checkpoints_folder_id
                            )

                        print(
                            f"[{clip}] OBB Detections: {total_detections} | Tracks: {tracker.next_object_id} | Homography failures: {homography_failures}"
                        )

                    except Exception as e:
                        # Non-strict fail: log warning, clean memory, and continue to next clip
                        print(
                            f"\nAdvertencia: Fallo procesando el clip {clip}. Se saltará este clip."
                        )
                        print(f"Detalle del error: {str(e)}")
                        sys.stdout.flush()
                        sys.stderr.flush()
                        continue

                    finally:
                        # FORCE RESOURCE RECOVERY
                        del tracker
                        del prev_gray
                        del prev_dets_poly
                        del frames_bgr
                        del frames_gray
                        del frame_indices
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # 6. DOWNLOAD MISSING FOR CONSOLIDATION: Ensuring all clips are in local dir
            if service:
                print(
                    "\nDownloading missing checkpoints from Google Drive for consolidation..."
                )
                self._download_missing_checkpoints(service, checkpoint_dir)

            # Merge checkpoints into final static_vehicles.json
            print("\nMerging clip checkpoints into final JSON mapping...")
            all_static_data = {}
            checkpoint_files = sorted(
                [f for f in os.listdir(checkpoint_dir) if f.endswith(".json")]
            )
            for fn in checkpoint_files:
                with open(os.path.join(checkpoint_dir, fn), "r") as f:
                    all_static_data.update(json.load(f))

            final_json_path = os.path.join(self.output_dir, "static_vehicles.json")
            with open(final_json_path, "w") as f:
                json.dump(all_static_data, f, indent=2)

            shutil.rmtree(checkpoint_dir)
            print(
                f"✓ Pipeline completed successfully. Local JSON saved to: {final_json_path}"
            )
            print(f"Total frames containing static vehicles: {len(all_static_data)}")

            # Upload consolidated JSON to Drive
            if service:
                self._upload_file_to_drive(service, final_json_path, self.folder_id)
                print("✓ Consolidated JSON uploaded to Google Drive.")

            # AUTO-DISCONNECT RUNTIME: Save compute credits in Google Colab on success
            try:
                from google.colab import runtime

                print(
                    "Desconectando sesión de Google Colab automáticamente para salvar créditos de cómputo..."
                )
                sys.stdout.flush()
                sys.stderr.flush()
                time.sleep(5)
                runtime.unassign()
            except ImportError:
                pass

            return final_json_path

        except Exception as e:
            print(f"\nError crítico en la ejecución del pipeline: {e}", file=sys.stderr)

            # Auto-disconnect Colab runtime to save compute credits on failures
            try:
                from google.colab import runtime

                print(
                    "Desconectando sesión de Google Colab debido a error para salvar créditos de cómputo..."
                )
                sys.stdout.flush()
                sys.stderr.flush()
                time.sleep(5)
                runtime.unassign()
            except ImportError:
                pass
            raise e


if __name__ == "__main__":
    # Minimal script usage trigger if called directly (no argparse)
    labeler = PseudoLabeler()
    labeler.run()
