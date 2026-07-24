"""Tests for the Base 2 classic-augmentation condition."""

from src.training.trainers.train_base_1 import Base1Trainer
from src.training.trainers.train_base_2 import Base2Trainer


def test_base2_changes_only_documented_augmentation_parameters() -> None:
    """Keep all Base 1 controls fixed except the Base 2 intervention."""
    base1 = Base1Trainer.DEFAULT_HYPERPARAMS
    base2 = Base2Trainer.DEFAULT_HYPERPARAMS
    changed_keys = {
        key for key in set(base1) | set(base2) if base1.get(key) != base2.get(key)
    }

    assert changed_keys == {
        "mosaic",
        "mixup",
        "copy_paste",
        "erasing",
        "close_mosaic",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    }
    assert base2["mosaic"] == 1.0
    assert base2["mixup"] == 0.15
    assert base2["copy_paste"] == 0.3
    assert base2["erasing"] == 0.4
    assert base2["close_mosaic"] == 10


def test_base2_uses_an_isolated_run_name() -> None:
    """Prevent Base 2 checkpoints from colliding with Base 1 artifacts."""
    trainer = Base2Trainer({"experiment_condition": Base2Trainer.EXPERIMENT_CONDITION})

    assert trainer.RUN_NAME == "base2"
    assert trainer.EXPERIMENT_CONDITION == "Base_2_Classic_Augmentation"
    assert trainer.get_hyperparameters()["mosaic"] == 1.0
