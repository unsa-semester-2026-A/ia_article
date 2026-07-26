"""Command-line orchestration for semantic-mask copy-paste augmentation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
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
from src.augmentation.metrics import ProductionMonitor
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
    slots = builder.collect_slots(
        Path(args.static_vehicles),
        train_clips,
        Path(args.labels_train),
        source_width=args.static_width,
        source_height=args.static_height,
        angles_in_radians=args.static_angles_in_radians,
    )
    if not slots:
        raise ValueError(
            "No safe static slots after coordinate conversion and overlap checks. "
            "Inspect static-slot resolution, angle unit, split, and labels before running SAM."
        )
    print(f"Collected {len(slots)} safe static slots before SAM extraction")
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
    archive_root = str(getattr(args, "archive_root", "")).strip("/")

    def archive_member(member: str) -> str:
        return f"{archive_root}/{member}" if archive_root else member

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in image_files:
            archive.write(path, archive_member(f"images/train/{path.name}"))
        for path in label_files:
            archive.write(path, archive_member(f"labels/train/{path.name}"))
        archive.write(manifest, archive_member("manifest.csv"))
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


def production(args: argparse.Namespace) -> int:
    """Run the complete train-only augmentation release with resumable evidence.

    The resulting release contains one shared synthetic delta.  Mejora B links
    it to the raw base and Mejora C links the identical delta to the LaMa base;
    duplicating 3+ GB base images would add no training information and makes
    validation of the two conditions less reliable.
    """
    output_root = Path(args.output_dir) / f"sam_copy_paste_{args.run_id}"
    output_root.mkdir(parents=True, exist_ok=True)
    workdir = output_root / "work"
    release = output_root / "release"
    dataset_release = release / f"sam_copy_paste_{args.run_id}"
    rendered = output_root / "rendered"
    metrics_path = release / "production_metrics.json"
    monitor = ProductionMonitor(metrics_path, output_root)
    monitor.start()
    started = time.monotonic()
    try:
        args.workdir = str(workdir)
        state_path = workdir / "augmentation_state.json"
        jobs_path = workdir / "jobs.jsonl"
        if args.resume and state_path.is_file() and jobs_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "prepared":
                raise ValueError(
                    f"Cannot resume state with status={state.get('status')}"
                )
            monitor.stage("prepare_resumed", jobs=int(state.get("jobs", 0)))
        else:
            prepare(args)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            monitor.stage(
                "prepare_complete",
                jobs=int(state["jobs"]),
                crop_tracks=int(state["crop_tracks"]),
                quotas=state["quotas"],
            )
        if not int(state.get("jobs", 0)):
            raise RuntimeError("Preparation produced zero jobs; refusing empty release")

        if args.resume and (rendered / "manifest.csv").is_file():
            monitor.stage("render_resumed")
        else:
            render(
                argparse.Namespace(
                    jobs_jsonl=str(jobs_path),
                    output_dir=str(rendered),
                    **_sync_args(args),
                )
            )
            monitor.stage("render_complete")

        _stage_dataset_release(rendered, dataset_release)
        validation = _validate_release(
            dataset_release, state, Path(args.split_metadata), jobs_path
        )
        release_manifest = {
            "run_id": args.run_id,
            "created_elapsed_seconds": round(time.monotonic() - started, 2),
            "policy": state["policy"],
            "real_counts": state["real_counts"],
            "track_counts": state["track_counts"],
            "quotas": state["quotas"],
            "validation": validation,
            "training_contract": {
                "delta_images": "images/train",
                "delta_labels": "labels/train",
                "validation_modified": False,
                "conditions": {
                    "mejora_b": "raw base + this delta",
                    "mejora_c": "LaMa base + this exact delta",
                },
            },
        }
        release_manifest_path = dataset_release / "release_manifest.json"
        release_manifest_path.write_text(
            json.dumps(release_manifest, indent=2), encoding="utf-8"
        )
        _write_release_readme(dataset_release, args.run_id)
        monitor.stage("release_validated", **validation)

        package_delta(
            argparse.Namespace(
                synthetic_images=str(dataset_release / "images" / "train"),
                synthetic_labels=str(dataset_release / "labels" / "train"),
                manifest=str(dataset_release / "manifest.csv"),
                output_dir=str(output_root),
                run_id=args.run_id,
                archive_root=dataset_release.name,
                **_sync_args(args),
            )
        )
        monitor.stage("training_delta_packaged")
        monitor.stop()
        audit_archive = output_root / f"sam_copy_paste_audit_{args.run_id}.zip"
        summary = {
            "run_id": args.run_id,
            "release_dir": str(dataset_release),
            "training_delta_zip": str(
                output_root / f"sam_copy_paste_delta_{args.run_id}.zip"
            ),
            "audit_zip": str(audit_archive),
            "metrics": str(metrics_path),
            "validation": validation,
        }
        (output_root / "production_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        _package_audit(output_root, workdir, release, dataset_release, args.run_id)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        monitor.stop(error=f"{type(exc).__name__}: {exc}")
        raise


def _sync_args(args: argparse.Namespace) -> dict[str, object]:
    """Return cloud-sync options required by nested command handlers."""
    return {
        "no_drive_sync": args.no_drive_sync,
        "token_path": args.token_path,
        "drive_results_folder_id": args.drive_results_folder_id,
        "drive_checkpoints_folder_id": args.drive_checkpoints_folder_id,
    }


def _stage_dataset_release(rendered: Path, dataset_release: Path) -> None:
    """Stage rendered files in the train-only directory shape consumed by YOLO."""
    source_pairs = (
        (rendered / "images", dataset_release / "images" / "train"),
        (rendered / "labels", dataset_release / "labels" / "train"),
    )
    for source, destination in source_pairs:
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if not path.is_file():
                continue
            target = destination / path.name
            if target.exists():
                continue
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
    for name in ("manifest.csv",):
        source = rendered / name
        target = dataset_release / name
        if not target.exists():
            shutil.copy2(source, target)


def _validate_release(
    release: Path,
    state: dict[str, object],
    metadata_path: Path,
    jobs_path: Path,
) -> dict[str, object]:
    """Verify image-label pairing and guarantee that output contains train only."""
    images = sorted((release / "images" / "train").glob("*.jpg"))
    labels = sorted((release / "labels" / "train").glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if not image_stems or image_stems != label_stems:
        raise ValueError(
            "Production release has missing or unmatched image/label pairs"
        )
    if (release / "images" / "val").exists() or (release / "labels" / "val").exists():
        raise ValueError("Synthetic release must not create a validation directory")
    train_clips = SyntheticDatasetBuilder(AugmentationConfig()).train_clip_ids(
        metadata_path
    )
    job_clips: set[str] = set()
    for line in jobs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            job_clips.add(str(json.loads(line)["background_clip_id"]))
    if not job_clips.issubset(train_clips):
        raise ValueError("A synthetic background is outside the train split")
    manifest_rows = {
        row["synthetic_id"]: row
        for row in csv.DictReader((release / "manifest.csv").open(encoding="utf-8"))
    }
    inserted_instances = 0
    for label in labels:
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 9:
                raise ValueError(f"Invalid YOLO-OBB line in generated label: {label}")
            values = [float(value) for value in fields[1:]]
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"Out-of-range generated OBB coordinate in {label}")
        if label.stem not in manifest_rows:
            raise ValueError(f"Generated label has no manifest row: {label.stem}")
        inserted_instances += len(json.loads(manifest_rows[label.stem]["objects"]))
    return {
        "synthetic_images": len(images),
        "synthetic_labels": len(labels),
        "inserted_instances": inserted_instances,
        "background_train_clips": len(job_clips),
        "validation_unchanged": True,
        "prepared_jobs": int(state["jobs"]),
    }


def _write_release_readme(release: Path, run_id: str) -> None:
    """Write trainer-facing instructions beside the uploaded release."""
    (release / "README.txt").write_text(
        "\n".join(
            [
                f"SAM copy-paste train-only release: {run_id}",
                "Use images/ and labels/ as augmentation_delta_* directories.",
                "Do not add these files to validation.",
                "Mejora B = raw base + this delta.",
                "Mejora C = LaMa base + this exact delta.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _package_audit(
    output_root: Path,
    workdir: Path,
    release: Path,
    dataset_release: Path,
    run_id: str,
) -> Path:
    """Bundle state, jobs, manifest and metrics separately from training data."""
    archive_path = output_root / f"sam_copy_paste_audit_{run_id}.zip"
    members = [
        workdir / "augmentation_state.json",
        workdir / "jobs.jsonl",
        dataset_release / "manifest.csv",
        dataset_release / "release_manifest.json",
        release / "production_metrics.json",
        dataset_release / "README.txt",
        output_root / f"sam_copy_paste_delta_{run_id}.quality_report.json",
        output_root / "production_summary.json",
    ]
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for member in members:
            if not member.is_file():
                raise FileNotFoundError(f"Missing audit artifact: {member}")
            archive.write(member, member.relative_to(output_root))
    return archive_path


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
                    "background_frame_id": frame_id,
                    "background_clip_id": slots_by_frame[frame_id][0].clip_id,
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
    prepare_parser.add_argument("--static-width", type=int, default=1920)
    prepare_parser.add_argument("--static-height", type=int, default=1080)
    prepare_parser.add_argument(
        "--static-angles-in-radians",
        action="store_true",
        default=True,
        help="Interpret static_vehicles.json OBB angles as radians (SMART default).",
    )
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
    production_parser = subparsers.add_parser("production", parents=[common])
    production_parser.add_argument("--split-metadata", required=True)
    production_parser.add_argument("--static-vehicles", required=True)
    production_parser.add_argument("--labels-train", required=True)
    production_parser.add_argument("--raw-images", required=True)
    production_parser.add_argument("--lama-images", required=True)
    production_parser.add_argument("--output-dir", required=True)
    production_parser.add_argument("--run-id", required=True)
    production_parser.add_argument("--sam-model", default="sam_b.pt")
    production_parser.add_argument("--static-width", type=int, default=1920)
    production_parser.add_argument("--static-height", type=int, default=1080)
    production_parser.add_argument(
        "--static-angles-in-radians", action="store_true", default=True
    )
    production_parser.add_argument("--max-source-tracks-per-class", type=int)
    production_parser.add_argument("--max-objects-per-class", type=int)
    production_parser.add_argument("--max-jobs", type=int)
    production_parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a prepared jobs manifest from the same run ID.",
    )
    production_parser.set_defaults(handler=production)
    return parser


def main() -> int:
    """Execute the selected pipeline stage."""
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - command-line wrapper
    raise SystemExit(main())
