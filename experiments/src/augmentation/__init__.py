"""Reproducible IC-Light synthetic augmentation for SMART training data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.augmentation.pipeline import AugmentationConfig, SyntheticDatasetBuilder

__all__ = ["AugmentationConfig", "SyntheticDatasetBuilder"]


def __getattr__(name: str):
    """Load the data pipeline only when its public types are requested.

    This keeps the IC-Light client importable in a minimal inference runtime,
    where pandas is intentionally not a dependency.
    """
    if name in __all__:
        from src.augmentation.pipeline import (
            AugmentationConfig,
            SyntheticDatasetBuilder,
        )

        return {
            "AugmentationConfig": AugmentationConfig,
            "SyntheticDatasetBuilder": SyntheticDatasetBuilder,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
