"""Command-line orchestration for the IC-Light synthetic dataset pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.augmentation.iclight import ICLightClient
from src.augmentation.pipeline import (
    OBB,
    AugmentationConfig,
    SyntheticDatasetBuilder,
    obb_to_yolo_line,
)
from src.augmentation.render import relight_variant, warp_crop_to_slot
from src.utils.io_manager import IOManager

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
    crops, real_counts = builder.extract_track_crops(
        Path(args.raw_images), Path(args.labels_train), train_clips, workdir / "crops"
    )
    slots = builder.collect_slots(
        Path(args.static_vehicles), train_clips, Path(args.labels_train)
    )
    track_counts: dict[int, int] = {}
    for crop in crops:
        track_counts[crop.class_id] = track_counts.get(crop.class_id, 0) + 1
    from src.augmentation.pipeline import quota_by_class

    quotas = quota_by_class(real_counts, track_counts, builder.config)
    jobs = []
    slots_iter = iter(slots)
    for class_id, quota in sorted(quotas.items()):
        class_crops = [crop for crop in crops if crop.class_id == class_id]
        for index in range(quota):
            try:
                slot = next(slots_iter)
            except StopIteration:
                break
            crop = class_crops[index % len(class_crops)]
            label_path = Path(args.labels_train) / f"{slot.frame_id}.txt"
            jobs.append(
                {
                    "synthetic_id": f"synth_{len(jobs):06d}",
                    "class_id": class_id,
                    "crop_path": crop.crop_path,
                    "target_points": slot.points,
                    "raw_background": str(
                        Path(args.raw_images) / f"{slot.frame_id}.jpg"
                    ),
                    "lama_background": str(
                        Path(args.lama_images) / f"{slot.frame_id}.jpg"
                    ),
                    "base_label_lines": label_path.read_text(
                        encoding="utf-8"
                    ).splitlines(),
                    "seed": args.seed + len(jobs),
                    "source_track_id": crop.track_id,
                    "slot_id": slot.slot_id,
                }
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
        "safe_slots": [slot.__dict__ for slot in slots],
        "status": "prepared",
    }
    state_path = workdir / "augmentation_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Prepared {len(slots)} safe static slots -> {state_path}")
    _sync_state(state_path, args)
    return 0


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


def render(args: argparse.Namespace) -> int:
    """Render prepared one-object jobs into matched Raw and LaMa variants.

    Jobs are JSONL records containing ``crop_path``, ``target_points``,
    ``raw_background``, ``lama_background``, ``synthetic_id`` and ``class_id``.
    A caller may add ``base_label_lines`` to preserve real objects already present
    in the selected background frame.
    """
    client = ICLightClient(args.iclight_url, args.iclight_api_name)
    jobs_path = Path(args.jobs_jsonl)
    output = Path(args.output_dir)
    rendered_rows: list[dict[str, object]] = []
    for raw_line in jobs_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        job = json.loads(raw_line)
        synthetic_id = str(job["synthetic_id"])
        foreground = warp_crop_to_slot(Path(job["crop_path"]), job["target_points"])
        raw_path = relight_variant(
            client,
            foreground,
            Path(job["raw_background"]),
            output / "raw/images" / f"{synthetic_id}.jpg",
            int(job["seed"]),
        )
        lama_path = relight_variant(
            client,
            foreground,
            Path(job["lama_background"]),
            output / "lama/images" / f"{synthetic_id}.jpg",
            int(job["seed"]),
        )
        obb = OBB(
            int(job["class_id"]), tuple(tuple(point) for point in job["target_points"])
        )
        lines = list(job.get("base_label_lines", [])) + [obb_to_yolo_line(obb)]
        for variant in ("raw", "lama"):
            labels_path = output / variant / "labels" / f"{synthetic_id}.txt"
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rendered_rows.append(
            {
                "synthetic_id": synthetic_id,
                "raw_image": str(raw_path),
                "lama_image": str(lama_path),
                "class_id": obb.class_id,
                "included": True,
                "quality_gate": "deferred_future_work",
                "seed": job["seed"],
            }
        )
    manifest = output / "manifest.csv"
    from src.augmentation.pipeline import write_manifest

    write_manifest(rendered_rows, manifest)
    _sync_state(manifest, args)
    print(f"Rendered {len(rendered_rows)} IC-Light jobs -> {output}")
    return 0


def _sync_state(path: Path, args: argparse.Namespace) -> None:
    """Upload resumable state to the dedicated augmentation checkpoint folder."""
    if args.no_drive_sync:
        return
    _drive(args).upload_file_to_drive(
        path, args.drive_checkpoints_folder_id, remote_name=path.name
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


def _drive(args: argparse.Namespace) -> IOManager:
    """Require a usable Drive connection before persisting production artifacts."""
    manager = IOManager(token_path=args.token_path, require_drive=True)
    if not manager.drive_service:
        raise RuntimeError(
            "Google Drive is unavailable; refusing to run without resumable artifacts."
        )
    return manager


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface shared by preparation and packaging."""
    parser = argparse.ArgumentParser(
        description="IC-Light synthetic augmentation pipeline"
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
    render_parser = subparsers.add_parser("render", parents=[common])
    render_parser.add_argument("--jobs-jsonl", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.add_argument("--iclight-url", default="http://127.0.0.1:7860")
    render_parser.add_argument("--iclight-api-name")
    render_parser.set_defaults(handler=render)
    return parser


def main() -> int:
    """Execute the selected pipeline stage."""
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover - command-line wrapper
    raise SystemExit(main())
