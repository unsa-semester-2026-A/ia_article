#!/usr/bin/env python3
"""Annotation parsing and statistical audit module for the SMART Challenge 2026.

This module parses the raw annotations CSV file, performs deterministic
clip-level splitting, validates class distribution parity, converts parametric
rotated bounding boxes to YOLO-OBB corner coordinates, and exports label files and
dataset metadata. It is highly optimized using multiprocessing and linear parsing
to save computational time.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Official class mapping from 1-indexed (CSV) to 0-indexed (YOLO)
CLASS_MAPPING: Dict[int, str] = {
    0: "auto",
    1: "combi",
    2: "microbus",
    3: "minibus",
    4: "omnibus",
    5: "articulado",
    6: "camion",
    7: "mototaxi",
    8: "motocicleta",
}

# Theoretical census statistics for strict dataset integrity validation
THEORETICAL_STATS: Dict[str, int] = {
    "total_frames": 54262,
    "total_objects": 601934,
    "total_clips": 1088,
    "empty_frames": 3394,
    "frames_1_obj": 6212,
    "frames_ge2_obj": 44656,
    "max_objs_per_frame": 53,
}

THEORETICAL_CLASSES: Dict[int, int] = {
    0: 481731,  # Car (auto)
    1: 10152,  # Combi
    2: 2802,  # Microbus
    3: 18941,  # Minibus
    4: 2283,  # Omnibus
    5: 250,  # Articulated bus
    6: 32668,  # Truck (camion)
    7: 5539,  # Mototaxi
    8: 47568,  # Motorcycle (motocicleta)
}


def parse_csv(csv_path: str) -> pd.DataFrame:
    """Reads the raw annotations CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the annotations CSV file.

    Returns:
        A pandas DataFrame containing raw CSV rows.

    Raises:
        FileNotFoundError: If the CSV file does not exist at csv_path.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file at: {csv_path}")
    return pd.read_csv(csv_path)


def convert_obb_to_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float, W: int, H: int
) -> List[float]:
    """Converts a parametric rotated bounding box to 4 normalized corners.

    Args:
        cx: Bbox center X.
        cy: Bbox center Y.
        w: Bbox width in pixels.
        h: Bbox height in pixels.
        angle_deg: Rotation angle in degrees, counter-clockwise.
        W: Image width.
        H: Image height.

    Returns:
        List of 8 floats representing corners in YOLO-OBB order:
        [x1, y1, x2, y2, x3, y3, x4, y4] in range [0.0, 1.0].
    """
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])

    x_rot = dx * cos_t - dy * sin_t + cx
    y_rot = dx * sin_t + dy * cos_t + cy

    x_norm = np.clip(x_rot / W, 0.0, 1.0)
    y_norm = np.clip(y_rot / H, 0.0, 1.0)

    return [
        float(x_norm[0]),
        float(y_norm[0]),
        float(x_norm[1]),
        float(y_norm[1]),
        float(x_norm[2]),
        float(y_norm[2]),
        float(x_norm[3]),
        float(y_norm[3]),
    ]


def process_single_row(
    row_tuple: Tuple[str, str, str, int, int],
) -> Tuple[int, List[int], int]:
    """Processes a single row from the CSV, parsing annotations and writing label file.

    Args:
        row_tuple: A tuple containing (frame_id, target, split, img_w, img_h).

    Returns:
        A tuple of (num_objects_parsed, list_of_class_ids, frame_objects_count).
    """
    frame_id, target, split, img_w, img_h = row_tuple

    # Output path in temporary system storage
    lbl_dir = f"/tmp/temp_labels/{split}"
    lbl_file = os.path.join(lbl_dir, f"{frame_id}.txt")

    if target == "none" or not target:
        # Write empty file for frames without objects
        with open(lbl_file, "w"):
            pass
        return 0, [], 0

    annotations = target.split(";")
    num_objects = 0
    classes_list: List[int] = []

    with open(lbl_file, "w") as f:
        for ann in annotations:
            ann_parts = ann.strip().split(" ")
            if len(ann_parts) != 6:
                continue

            class_id = int(ann_parts[0]) - 1  # 0-index conversion
            cx = float(ann_parts[1])
            cy = float(ann_parts[2])
            width = float(ann_parts[3])
            height = float(ann_parts[4])
            angle_deg = float(ann_parts[5])

            corners = convert_obb_to_corners(
                cx, cy, width, height, angle_deg, img_w, img_h
            )
            corners_str = " ".join([f"{c:.6f}" for c in corners])
            f.write(f"{class_id} {corners_str}\n")

            num_objects += 1
            classes_list.append(class_id)

    return num_objects, classes_list, num_objects


def main() -> None:
    """CLI execution entry point with highly optimized multiprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Fase 0: Optimized Parser and YOLO-OBB Data Preparation."
    )
    parser.add_argument(
        "--csv",
        default="/content/drive/MyDrive/ia_article/00_raw/train.csv",
        help="Path to annotations train.csv file.",
    )
    parser.add_argument(
        "--output_dir",
        default="/content/drive/MyDrive/ia_article/01_processed",
        help="Target folder to save generated outputs.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion of training clips (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic split random seed (default: 42).",
    )
    parser.add_argument(
        "--img_w",
        type=int,
        default=1920,
        help="Default width of images (default: 1920).",
    )
    parser.add_argument(
        "--img_h",
        type=int,
        default=1080,
        help="Default height of images (default: 1080).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel worker processes (default: 2 for Colab).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize clean temporary directories for fast I/O
    os.makedirs("/tmp/temp_labels/train", exist_ok=True)
    os.makedirs("/tmp/temp_labels/val", exist_ok=True)

    try:
        print("Loading annotations dataset...")
        df_raw = parse_csv(args.csv)

        # 1. Deterministic split by clip_id
        print("Computing deterministic splits...")
        df_raw["clip_id"] = df_raw["Id"].apply(lambda x: "_".join(x.split("_")[:-1]))
        unique_clips = sorted(df_raw["clip_id"].unique())

        from sklearn.model_selection import train_test_split

        train_clips, val_clips = train_test_split(
            unique_clips, train_size=args.train_ratio, random_state=args.seed
        )

        train_clips_set = set(train_clips)
        clip_to_split = {
            clip: "train" if clip in train_clips_set else "val" for clip in unique_clips
        }

        # 2. Export metadata and configuration files
        metadata_records = [
            {"clip_id": clip, "split": split} for clip, split in clip_to_split.items()
        ]
        metadata_df = pd.DataFrame(metadata_records)
        metadata_path = os.path.join(args.output_dir, "split_metadata.csv")
        metadata_df.to_csv(metadata_path, index=False)
        print(f"✓ Split metadata CSV exported to: {metadata_path}")

        # Write smart_dataset.yaml
        yaml_content = {
            "path": "/content/dataset",
            "train": "train/images",
            "val": "val/images",
            "names": CLASS_MAPPING,
        }
        yaml_path = os.path.join(args.output_dir, "smart_dataset.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Configuration YAML written to: {yaml_path}")

        # Prepare parameters list for multiprocessing
        print("Preparing tasks for parallel execution...")
        tasks = []
        for _, row in df_raw.iterrows():
            frame_id = str(row["Id"])
            target = "" if pd.isna(row["Target"]) else str(row["Target"])
            clip_id = "_".join(frame_id.split("_")[:-1])
            split = clip_to_split[clip_id]
            tasks.append((frame_id, target, split, args.img_w, args.img_h))

        # 3. Multiprocessing Pool for parsing and file writing
        print(
            f"Executing parsing and OBB corner calculation "
            f"using parallel workers ({args.workers})..."
        )
        total_objects = 0
        total_frames = len(tasks)
        empty_frames = 0
        frames_1_obj = 0
        frames_ge2_obj = 0
        max_objs_per_frame = 0

        class_counts = {cls: 0 for cls in range(9)}
        split_class_counts: Dict[str, Dict[int, int]] = {
            "train": {cls: 0 for cls in range(9)},
            "val": {cls: 0 for cls in range(9)},
        }
        split_total_objs = {"train": 0, "val": 0}

        # Run process pool executor
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(process_single_row, tasks))

        # 4. Aggregating statistics from workers
        print("Aggregating statistics...")
        for i, (num_objs, classes, frame_count) in enumerate(results):
            task_split = tasks[i][2]

            total_objects += num_objs
            split_total_objs[task_split] += num_objs

            for cls in classes:
                class_counts[cls] += 1
                split_class_counts[task_split][cls] += 1

            if frame_count == 0:
                empty_frames += 1
            elif frame_count == 1:
                frames_1_obj += 1
            else:
                frames_ge2_obj += 1

            if frame_count > max_objs_per_frame:
                max_objs_per_frame = frame_count

        # 5. Strict Audit Statistics Validation
        print("\n=== Performing Strict Statistical Audit ===")
        print(f"Total Frames: {total_frames} / {THEORETICAL_STATS['total_frames']}")
        assert total_frames == THEORETICAL_STATS["total_frames"], (
            "Frame count mismatch."
        )

        print(
            f"Total OBB Objects: {total_objects} / {THEORETICAL_STATS['total_objects']}"
        )
        assert total_objects == THEORETICAL_STATS["total_objects"], (
            "Object count mismatch."
        )

        print(
            f"Total Unique Clips: {len(unique_clips)} / "
            f"{THEORETICAL_STATS['total_clips']}"
        )
        assert len(unique_clips) == THEORETICAL_STATS["total_clips"], (
            "Clip count mismatch."
        )

        print(
            f"Empty Frames ('none'): {empty_frames} / "
            f"{THEORETICAL_STATS['empty_frames']}"
        )
        assert empty_frames == THEORETICAL_STATS["empty_frames"], (
            "Empty frames count mismatch."
        )

        print(
            f"Frames with 1 object: {frames_1_obj} / "
            f"{THEORETICAL_STATS['frames_1_obj']}"
        )
        assert frames_1_obj == THEORETICAL_STATS["frames_1_obj"], (
            "Frames with 1 object mismatch."
        )

        print(
            f"Frames with >=2 objects: {frames_ge2_obj} / "
            f"{THEORETICAL_STATS['frames_ge2_obj']}"
        )
        assert frames_ge2_obj == THEORETICAL_STATS["frames_ge2_obj"], (
            "Frames with >=2 objects mismatch."
        )

        print(
            f"Max objects in single frame: {max_objs_per_frame} / "
            f"{THEORETICAL_STATS['max_objs_per_frame']}"
        )
        assert max_objs_per_frame == THEORETICAL_STATS["max_objs_per_frame"], (
            "Max objects per frame mismatch."
        )

        print("\nClass Distribution:")
        for cls_id, target in THEORETICAL_CLASSES.items():
            count = class_counts.get(cls_id, 0)
            pct = (count / total_objects) * 100
            print(
                f"  Class {cls_id} ({CLASS_MAPPING[cls_id]}): "
                f"{count:6d} ({pct:5.2f}%) [Target: {target}]"
            )
            assert count == target, f"Instance count mismatch for Class {cls_id}."

        print("✓ Strict statistical audit passed successfully.")

        # 6. Verify class distribution parity in splits
        print("\n=== Verifying Class Distribution Parity ===")
        for cls in range(9):
            pct_total = (class_counts[cls] / total_objects) * 100
            pct_train = (
                split_class_counts["train"][cls] / split_total_objs["train"]
            ) * 100
            pct_val = (split_class_counts["val"][cls] / split_total_objs["val"]) * 100

            diff_train = abs(pct_train - pct_total)
            diff_val = abs(pct_val - pct_total)

            print(f"Class {cls} ({CLASS_MAPPING[cls]}):")
            print(f"  Train: {pct_train:6.2f}% (diff: {diff_train:5.2f}%)")
            print(f"  Val:   {pct_val:6.2f}% (diff: {diff_val:5.2f}%)")

            assert diff_train <= 2.0, (
                f"Class {cls} train split deviation exceeds 2% absolute."
            )
            assert diff_val <= 2.0, (
                f"Class {cls} val split deviation exceeds 2% absolute."
            )
        print("✓ Split parity checks passed successfully.")

        # 7. Compress the generated labels
        zip_output_path = os.path.join(args.output_dir, "yolo_obb_labels")
        print(f"\nArchiving labels into: {zip_output_path}.zip...")
        shutil.make_archive(zip_output_path, "zip", "/tmp/temp_labels")
        print("✓ Label archiving completed successfully.")

        print("\n✓ Full data preparation phase completed successfully.")

    except Exception as e:
        print(
            f"\nError during data preparation execution: {e}",
            file=sys.stderr,
        )
        # Clean up temporary folders since we are killing the runtime immediately
        if os.path.exists("/tmp/temp_labels"):
            shutil.rmtree("/tmp/temp_labels")

        # Disconnect and terminate the Colab environment to save credits
        try:
            from google.colab import runtime

            runtime.unassign()
        except ImportError:
            # We are not in Google Colab (e.g. running locally or on Kaggle)
            sys.exit(1)
        sys.exit(1)

    finally:
        # Clean up temporary system folders (for normal execution)
        if os.path.exists("/tmp/temp_labels"):
            shutil.rmtree("/tmp/temp_labels")


if __name__ == "__main__":
    main()
