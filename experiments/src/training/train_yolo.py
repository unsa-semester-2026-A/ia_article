"""YOLO OBB Training Orchestrator Module.

This script runs training for YOLO26s-OBB under specified experimental conditions,
supporting configuration files, intermediate checkpointing, and Google Drive upload
for persistent results.
"""

import argparse
import os
import shutil
import sys
from typing import Any, Dict

import yaml

# Import Google Drive utilities
from src.utils.drive import (
    find_or_create_folder,
    get_drive_service,
    get_project_root_folder_id,
    upload_file_to_drive,
)
from ultralytics import YOLO


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML training configuration file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dict: Hyperparameters and training settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def main() -> None:
    """Entry point for parsing arguments and launching YOLO training."""
    parser = argparse.ArgumentParser(
        description="Fase 3: Train YOLO OBB under specified condition."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the training configuration YAML file (e.g. configs/base1.yaml).",
    )
    parser.add_argument(
        "--data",
        default="smart_dataset.yaml",
        help="Path to the dataset configuration YAML file.",
    )
    parser.add_argument(
        "--project",
        default="runs/obb",
        help="Directory to save training run logs and weights.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Name of the training run. Defaults to the config filename without extension.",
    )
    parser.add_argument(
        "--token",
        default="/content/drive/MyDrive/ia_article/token/token.json",
        help="Path to the Google Drive auth token JSON.",
    )
    parser.add_argument(
        "--known_folder_id",
        default="1J5ogC3q6jyYlk3wuYyxpYZHslUg6eGtN",
        help="ID of the 02_pseudo_labeling folder in Google Drive.",
    )
    parser.add_argument(
        "--local_drive_dir",
        default="/content/drive/MyDrive/ia_article",
        help="Local mount directory for Google Drive (if mounted).",
    )

    args = parser.parse_args()

    # Determine run name from config filename if not specified
    run_name = args.name
    if not run_name:
        run_name = os.path.splitext(os.path.basename(args.config))[0]

    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Resolve model file
    model_name = config.get("model", "yolo26s-obb.pt")
    print(f"Initializing model {model_name}...")
    model = YOLO(model_name)

    # Build train arguments
    # Filter out model from train args as we pass it to initializer
    train_args = {k: v for k, v in config.items() if k != "model"}

    # Set default values if not specified in config
    train_args.setdefault("imgsz", 640)
    train_args.setdefault("epochs", 100)
    train_args.setdefault("batch", 16)
    train_args.setdefault("patience", 20)
    train_args.setdefault("seed", 42)

    # Save checkpoints every 10 epochs
    train_args["save_period"] = 10

    # Device configuration
    if "device" not in train_args:
        import torch

        train_args["device"] = 0 if torch.cuda.is_available() else "cpu"

    print("\nTraining configuration parameters:")
    for k, v in sorted(train_args.items()):
        print(f"  {k}: {v}")

    print(f"\nStarting training run '{run_name}' on dataset '{args.data}'...")
    results = model.train(
        data=args.data, project=args.project, name=run_name, **train_args
    )

    # Locate results
    # Use save_dir from results object or build default fallback
    save_dir = getattr(results, "save_dir", os.path.join(args.project, run_name))
    print(f"\nTraining completed. Results saved locally at: {save_dir}")

    best_weights = os.path.join(save_dir, "weights", "best.pt")
    last_weights = os.path.join(save_dir, "weights", "last.pt")
    results_csv = os.path.join(save_dir, "results.csv")

    # Rename file targets for Google Drive destination
    dest_best_name = f"best_{run_name}.pt"
    dest_last_name = f"last_{run_name}.pt"
    dest_results_name = f"results_{run_name}.csv"

    # Persistent Storage Sync
    # Path 1: Google Drive is mounted as a local directory
    local_target_dir = os.path.join(args.local_drive_dir, "03_models")
    if os.path.exists(args.local_drive_dir):
        print(
            f"✓ Local Google Drive mount detected at: {args.local_drive_dir}. Copying files..."
        )
        try:
            os.makedirs(local_target_dir, exist_ok=True)

            if os.path.exists(best_weights):
                dest_path = os.path.join(local_target_dir, dest_best_name)
                shutil.copy(best_weights, dest_path)
                print(f"✓ Copied best weights to: {dest_path}")

            if os.path.exists(last_weights):
                dest_path = os.path.join(local_target_dir, dest_last_name)
                shutil.copy(last_weights, dest_path)
                print(f"✓ Copied last weights to: {dest_path}")

            if os.path.exists(results_csv):
                dest_path = os.path.join(local_target_dir, dest_results_name)
                shutil.copy(results_csv, dest_path)
                print(f"✓ Copied results CSV to: {dest_path}")

            print("✓ Files successfully copied to mounted Google Drive.")
            return
        except Exception as e:
            print(f"Failed to copy files to mounted Google Drive: {e}", file=sys.stderr)
            print("Attempting API upload fallback...")

    # Path 2: Google Drive API authentication and upload
    print("Initiating Google Drive API upload...")
    service = get_drive_service(args.token)
    if not service:
        print(
            "Warning: Could not initialize Google Drive service. "
            "Please download weights manually from the local run directory.",
            file=sys.stderr,
        )
        return

    # Find the root project folder
    root_id = get_project_root_folder_id(service, args.known_folder_id)
    if not root_id:
        print(
            "Error: Could not resolve root folder ID from Google Drive. "
            "Skipping Drive API upload.",
            file=sys.stderr,
        )
        return

    # Find or create '03_models' folder
    models_folder_id = find_or_create_folder(service, "03_models", root_id)
    if not models_folder_id:
        print(
            "Error: Could not resolve or create '03_models' folder on Google Drive. "
            "Skipping API upload.",
            file=sys.stderr,
        )
        return

    # Upload files using API
    if os.path.exists(best_weights):
        # Create a temp copy with renamed filename for correct Drive filename
        temp_best = os.path.join(save_dir, "weights", dest_best_name)
        shutil.copy(best_weights, temp_best)
        print(f"Uploading {dest_best_name} to Google Drive...")
        file_id = upload_file_to_drive(
            service, temp_best, models_folder_id, "application/octet-stream"
        )
        if file_id:
            print(f"✓ Successfully uploaded {dest_best_name} (ID: {file_id})")
        # Clean up temp copy
        if os.path.exists(temp_best):
            os.remove(temp_best)

    if os.path.exists(last_weights):
        temp_last = os.path.join(save_dir, "weights", dest_last_name)
        shutil.copy(last_weights, temp_last)
        print(f"Uploading {dest_last_name} to Google Drive...")
        file_id = upload_file_to_drive(
            service, temp_last, models_folder_id, "application/octet-stream"
        )
        if file_id:
            print(f"✓ Successfully uploaded {dest_last_name} (ID: {file_id})")
        if os.path.exists(temp_last):
            os.remove(temp_last)

    if os.path.exists(results_csv):
        temp_results = os.path.join(save_dir, dest_results_name)
        shutil.copy(results_csv, temp_results)
        print(f"Uploading {dest_results_name} to Google Drive...")
        file_id = upload_file_to_drive(
            service, temp_results, models_folder_id, "text/csv"
        )
        if file_id:
            print(f"✓ Successfully uploaded {dest_results_name} (ID: {file_id})")
        if os.path.exists(temp_results):
            os.remove(temp_results)

    print("✓ Google Drive API uploads completed successfully.")


if __name__ == "__main__":
    main()
