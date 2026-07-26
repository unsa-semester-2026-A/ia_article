"""Command-line orchestration for semantic-mask copy-paste augmentation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from src.augmentation.copy_paste import (
    background_is_unchanged,
    composite_semantic_foreground,
)
from src.augmentation.masking import SamBoxMasker
from src.augmentation.pipeline import (
    OBB,
    AugmentationConfig,
    SyntheticDatasetBuilder,
    obb_to_yolo_line,
)
from src.augmentation.render import warp_crop_to_slot

RESULTS_FOLDER_ID = "1kZfwzClMDM3Ci854IV8FpobTGd4fXQTX"
CHECKPOINTS_FOLDER_ID = "1XANPylHPW5sa0cBJgHRRVrQHu8-j7PWN"


def _config(args: argparse.Namespace) -> AugmentationConfig:
    """Construct the selected final policy from command-line values."""
    return AugmentationConfig(
        tau=args.tau,
        reuse_cap=args.reuse_cap,
        budget_fraction=args.budget_fraction,
        seed=args.seed,
    )


def prepare(args: argparse.Namespace) -> int:
    """Validate inputs, materialize safe slots and persist an immutable run state.

    Crop extraction and rendering are intentionally separate from this command so
    the expensive Colab session can resume from ``augmentation_state.json``.
    """
    builder = SyntheticDatasetBuilder(_config(args))
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    train_clips = builder.train_clip_ids(Path(args.split_metadata))
    source_limits = args.max_source_tracks_per_class
    if source_limits is None:
        source_limits = _production_source_limits(
            _count_real_instances(Path(args.labels_train), train_clips),
            builder.config,
        )
    crops, real_counts, mask_rejections = builder.extract_track_crops(
        Path(args.raw_images),
        Path(args.labels_train),
        train_clips,
        workdir / "crops",
        masker=SamBoxMasker(args.sam_model),
        max_tracks_per_class=source_limits,
    )
    slots = builder.collect_slots(
        Path(args.static_vehicles), train_clips, Path(args.labels_train)
    )
    track_counts: dict[int, int] = {}
    for crop in crops:
        track_counts[crop.class_id] = track_counts.get(crop.class_id, 0) + 1
    from src.augmentation.pipeline import quota_by_class

    quotas = quota_by_class(real_counts, track_counts, builder.config)
    if args.max_objects_per_class is not None:
        quotas = {
            class_id: min(quota, args.max_objects_per_class)
            for class_id, quota in quotas.items()
        }
    jobs = _build_grouped_jobs(
        crops=crops,
        slots=slots,
        quotas=quotas,
        labels_dir=Path(args.labels_train),
        lama_images=Path(args.lama_images),
        reuse_cap=builder.config.reuse_cap,
        seed=args.seed,
        max_objects_per_image=builder.config.max_objects_per_image,
        max_jobs=args.max_jobs,
    )
    jobs_path = workdir / "jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8"
    )
    state = {
        "policy": asdict(builder.config),
        "train_clips": len(train_clips),
        "crop_tracks": len(crops),
        "real_counts": real_counts,
        "track_counts": track_counts,
        "quotas": quotas,
        "jobs": len(jobs),
        "mask_rejections": mask_rejections,
        "source_track_limits": source_limits,
        "safe_slots": [slot.__dict__ for slot in slots],
        "status": "prepared",
    }
    state_path = workdir / "augmentation_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Prepared {len(slots)} safe static slots -> {state_path}")
    _sync_state(state_path, args)
    return 0


def _count_real_instances(labels_dir: Path, train_clips: set[str]) -> dict[int, int]:
    """Count train instances cheaply before allocating expensive SAM calls."""
    from src.augmentation.pipeline import load_yolo_obb, split_frame_id

    counts: dict[int, int] = {}
    for label_path in labels_dir.glob("*.txt"):
        clip_id, _ = split_frame_id(label_path.stem)
        if clip_id not in train_clips:
            continue
        for box in load_yolo_obb(label_path):
            counts[box.class_id] = counts.get(box.class_id, 0) + 1
    return counts


def _production_source_limits(
    real_counts: dict[int, int], config: AugmentationConfig
) -> dict[int, int]:
    """Bound SAM work to the tracks needed by policy plus a rejection reserve."""
    total = sum(real_counts.values())
    target = math.ceil(config.tau * total)
    limits: dict[int, int] = {}
    for class_id, real_count in real_counts.items():
        if not total or real_count / total >= config.tau:
            limits[class_id] = 0
            continue
        desired = min(real_count, max(0, target - real_count))
        # Two candidates per minimally required source track leave room for
        # semantic-mask rejections without scanning every temporal track.
        limits[class_id] = max(1, math.ceil(desired / config.reuse_cap) * 2)
    return limits


def package(args: argparse.Namespace) -> int:
    """Create and persist the two complete Raw/LaMa synthetic training ZIPs."""
    builder = SyntheticDatasetBuilder(_config(args))
    output = Path(args.output_dir)
    manifest = Path(args.manifest)
    raw_report = builder.package_full_dataset(
        Path(args.raw_images),
        Path(args.base_labels),
        Path(args.synthetic_raw_images),
        Path(args.synthetic_labels),
        manifest,
        output / "smart_raw_synthetic_train.zip",
        "raw",
    )
    lama_report = builder.package_full_dataset(
        Path(args.lama_images),
        Path(args.base_labels),
        Path(args.synthetic_lama_images),
        Path(args.synthetic_labels),
        manifest,
        output / "smart_lama_synthetic_train.zip",
        "lama",
    )
    report_path = output / "augmentation_package_report.json"
    report_path.write_text(
        json.dumps({"raw": raw_report, "lama": lama_report}, indent=2), encoding="utf-8"
    )
    _sync_file(output / "smart_raw_synthetic_train.zip", args)
    _sync_file(output / "smart_lama_synthetic_train.zip", args)
    _sync_file(report_path, args)
    print(json.dumps({"raw": raw_report, "lama": lama_report}, indent=2))
    return 0


def package_delta(args: argparse.Namespace) -> int:
    """Package only synthetic files so Kaggle need not duplicate base data."""
    import hashlib
    import zipfile

    output = Path(args.output_dir)
    images = Path(args.synthetic_images)
    labels = Path(args.synthetic_labels)
    manifest = Path(args.manifest)
    image_files = sorted(path for path in images.glob("*.jpg"))
    label_files = sorted(labels.glob("*.txt"))
    if {path.stem for path in image_files} != {path.stem for path in label_files}:
        raise ValueError("Delta image/label stems do not match")
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f"sam_copy_paste_delta_{args.run_id}.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in image_files:
            archive.write(path, f"images/train/{path.name}")
        for path in label_files:
            archive.write(path, f"labels/train/{path.name}")
        archive.write(manifest, "manifest.csv")
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report = {
        "run_id": args.run_id,
        "archive": archive_path.name,
        "sha256": digest.hexdigest(),
        "synthetic_images": len(image_files),
        "synthetic_labels": len(label_files),
    }
    report_path = output / f"sam_copy_paste_delta_{args.run_id}.quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _sync_file(archive_path, args)
    _sync_file(report_path, args)
    print(json.dumps(report, indent=2))
    return 0


def render(args: argparse.Namespace) -> int:
    """Render grouped semantic crops onto LaMa backgrounds without generation."""
    jobs_path = Path(args.jobs_jsonl)
    output = Path(args.output_dir)
    rendered_rows: list[dict[str, object]] = []
    for raw_line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        job = json.loads(raw_line)
        synthetic_id = str(job["synthetic_id"])
        background = _read_image(Path(job["lama_background"]))
        composite = background.copy()
        lines = list(job.get("base_label_lines", []))
        object_rows = []
        for object_job in job["objects"]:
            foreground = warp_crop_to_slot(
                Path(object_job["crop_path"]), object_job["target_points"]
            )
            previous = composite.copy()
            composite = composite_semantic_foreground(composite, foreground)
            if not background_is_unchanged(previous, composite, foreground[:, :, 3]):
                raise RuntimeError(
                    "Copy-paste altered pixels outside its semantic mask"
                )
            obb = OBB(
                int(object_job["class_id"]),
                tuple(tuple(point) for point in object_job["target_points"]),
            )
            lines.append(obb_to_yolo_line(obb))
            object_rows.append(
                {
                    "class_id": obb.class_id,
                    "source_track_id": object_job["source_track_id"],
                    "slot_id": object_job["slot_id"],
                }
            )
        image_path = output / "images" / f"{synthetic_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write synthetic image: {image_path}")
        labels_path = output / "labels" / f"{synthetic_id}.txt"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rendered_rows.append(
            {
                "synthetic_id": synthetic_id,
                "image": str(image_path),
                "objects": json.dumps(object_rows),
                "included": True,
                "quality_gate": "semantic_mask_and_background_passthrough",
                "seed": job["seed"],
            }
        )
    manifest = output / "manifest.csv"
    from src.augmentation.pipeline import write_manifest

    write_manifest(rendered_rows, manifest)
    _sync_state(manifest, args)
    print(f"Rendered {len(rendered_rows)} semantic copy-paste jobs -> {output}")
    return 0


def _build_grouped_jobs(
    *,
    crops: list[object],
    slots: list[object],
    quotas: dict[int, int],
    labels_dir: Path,
    lama_images: Path,
    reuse_cap: int,
    seed: int,
    max_objects_per_image: int,
    max_jobs: int | None = None,
) -> list[dict[str, object]]:
    """Pack quota-limited objects into backgrounds with up to three safe slots."""
    by_class: dict[int, list[object]] = {}
    for crop in crops:
        by_class.setdefault(crop.class_id, []).append(crop)
    slots_by_frame: dict[str, list[object]] = {}
    for slot in slots:
        slots_by_frame.setdefault(slot.frame_id, []).append(slot)
    remaining = {class_id: quota for class_id, quota in quotas.items() if quota > 0}
    source_index = {class_id: 0 for class_id in remaining}
    jobs: list[dict[str, object]] = []
    for frame_id, frame_slots in sorted(slots_by_frame.items()):
        if max_jobs is not None and len(jobs) >= max_jobs:
            break
        while frame_slots and remaining:
            if max_jobs is not None and len(jobs) >= max_jobs:
                break
            non_overlapping_slots = []
            for candidate in frame_slots:
                candidate_box = OBB(-1, tuple(candidate.points))
                if all(
                    _obb_overlap(candidate_box, OBB(-1, tuple(chosen.points))) == 0
                    for chosen in non_overlapping_slots
                ):
                    non_overlapping_slots.append(candidate)
                if len(non_overlapping_slots) == max_objects_per_image:
                    break
            if not non_overlapping_slots:
                break
            class_ids = sorted(remaining, key=lambda value: (-remaining[value], value))[
                : len(non_overlapping_slots)
            ]
            class_ids = [class_id for class_id in class_ids if by_class.get(class_id)]
            if not class_ids:
                break
            objects = []
            selected_slots = non_overlapping_slots[: len(class_ids)]
            for class_id, slot in zip(class_ids, selected_slots, strict=True):
                index = source_index[class_id]
                crop = by_class[class_id][index // reuse_cap]
                source_index[class_id] += 1
                remaining[class_id] -= 1
                if remaining[class_id] == 0:
                    del remaining[class_id]
                objects.append(
                    {
                        "class_id": class_id,
                        "crop_path": crop.crop_path,
                        "target_points": slot.points,
                        "source_track_id": crop.track_id,
                        "slot_id": slot.slot_id,
                    }
                )
            label_path = labels_dir / f"{frame_id}.txt"
            jobs.append(
                {
                    "synthetic_id": f"synth_{len(jobs):06d}",
                    "objects": objects,
                    "lama_background": str(lama_images / f"{frame_id}.jpg"),
                    "base_label_lines": label_path.read_text(
                        encoding="utf-8"
                    ).splitlines(),
                    "seed": seed + len(jobs),
                }
            )
            selected_ids = {slot.slot_id for slot in selected_slots}
            frame_slots = [
                slot for slot in frame_slots if slot.slot_id not in selected_ids
            ]
    return jobs


def _obb_overlap(first: OBB, second: OBB) -> float:
    """Return convex-polygon overlap without importing a second geometry module."""
    area, _ = cv2.intersectConvexConvex(
        np.asarray(first.points, dtype=np.float32),
        np.asarray(second.points, dtype=np.float32),
    )
    return float(max(area, 0.0))


def _read_image(path: Path):
    """Load one BGR image or fail with an actionable path."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load LaMa background: {path}")
    return image


def _sync_state(path: Path, args: argparse.Namespace) -> None:
    """Upload resumable state to the dedicated augmentation checkpoint folder."""
    if args.no_drive_sync:
        return
    _drive(args).upload_file_to_drive(
        path,
        args.drive_checkpoints_folder_id,
        remote_name=f"{path.parent.name}_{path.name}",
    )


def _sync_file(path: Path, args: argparse.Namespace) -> None:
    """Upload an immutable output under its explicit, non-colliding filename."""
    if args.no_drive_sync:
        return
    _drive(args).upload_file_to_drive(
        path,
        args.drive_results_folder_id,
        mime_type="application/zip" if path.suffix == ".zip" else "application/json",
        remote_name=path.name,
    )


def _drive(args: argparse.Namespace) -> Any:
    """Require a usable Drive connection before persisting production artifacts."""
    from src.utils.io_manager import IOManager

    manager = IOManager(token_path=args.token_path, require_drive=True)
    if not manager.drive_service:
        raise RuntimeError(
            "Google Drive is unavailable; refusing to run without resumable artifacts."
        )
    return manager


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface shared by preparation and packaging."""
    parser = argparse.ArgumentParser(
        description="SAM semantic-mask copy-paste augmentation pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tau", type=float, default=0.02)
    common.add_argument("--reuse-cap", type=int, default=10)
    common.add_argument("--budget-fraction", type=float, default=1.0)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument(
        "--token-path", default="/content/drive/MyDrive/ia_article/token/token.json"
    )
    common.add_argument("--drive-results-folder-id", default=RESULTS_FOLDER_ID)
    common.add_argument("--drive-checkpoints-folder-id", default=CHECKPOINTS_FOLDER_ID)
    common.add_argument("--no-drive-sync", action="store_true")
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--split-metadata", required=True)
    prepare_parser.add_argument("--static-vehicles", required=True)
    prepare_parser.add_argument("--labels-train", required=True)
    prepare_parser.add_argument("--raw-images", required=True)
    prepare_parser.add_argument("--lama-images", required=True)
    prepare_parser.add_argument("--workdir", required=True)
    prepare_parser.add_argument("--sam-model", default="sam_b.pt")
    prepare_parser.add_argument(
        "--max-source-tracks-per-class",
        type=int,
        help="Bound SAM calls per class; required for a fast smoke/demo run.",
    )
    prepare_parser.add_argument(
        "--max-objects-per-class",
        type=int,
        help="Cap rendered additions per class (for example 2 in the demo).",
    )
    prepare_parser.add_argument(
        "--max-jobs", type=int, help="Cap rendered backgrounds for a smoke/demo run."
    )
    prepare_parser.set_defaults(handler=prepare)
    package_parser = subparsers.add_parser("package", parents=[common])
    package_parser.add_argument("--raw-images", required=True)
    package_parser.add_argument("--lama-images", required=True)
    package_parser.add_argument("--base-labels", required=True)
    package_parser.add_argument("--synthetic-raw-images", required=True)
    package_parser.add_argument("--synthetic-lama-images", required=True)
    package_parser.add_argument("--synthetic-labels", required=True)
    package_parser.add_argument("--manifest", required=True)
    package_parser.add_argument("--output-dir", required=True)
    package_parser.set_defaults(handler=package)
    delta_parser = subparsers.add_parser("package-delta", parents=[common])
    delta_parser.add_argument("--synthetic-images", required=True)
    delta_parser.add_argument("--synthetic-labels", required=True)
    delta_parser.add_argument("--manifest", required=True)
    delta_parser.add_argument("--output-dir", required=True)
    delta_parser.add_argument("--run-id", required=True)
    delta_parser.set_defaults(handler=package_delta)
    render_parser = subparsers.add_parser("render", parents=[common])
    render_parser.add_argument("--jobs-jsonl", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.set_defaults(handler=render)
    return parser


def main() -> int:
    """Execute the selected pipeline stage."""
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - command-line wrapper
    raise SystemExit(main())
