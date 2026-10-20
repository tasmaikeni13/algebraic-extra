"""ARDS (Algebraic Rational Decay Schedule) Pipelines and Configurations.

This module provides dedicated, single-seed (seed 42) training and evaluation
pipelines for Standard and Pure Algebraic Transformers at the 125M parameter scale.
Cosine learning rate schedules are strictly prohibited to ensure algebraic purity.
"""

from ards.configs import (
    ARDS_PRESETS,
    ARDSTrainingConfig,
    algebraic_ards,
    get_ards_config,
    get_training_config,
    standard_ards,
    validate_schedule_not_cosine,
)
from ards.evaluate import (
    evaluate_ards_run,
    evaluate_main,
    evaluate_recall,
    evaluate_zero_shot,
    run_ards_evaluation,
    validate_ards_checkpoint_purity,
)
from ards.schedule import (
    DEFAULT_125M_PEAK_LR,
    DEFAULT_125M_TOTAL_STEPS,
    DEFAULT_125M_WARMUP_STEPS,
    DEFAULT_SCALE_FACTOR,
    FORBIDDEN_COSINE_KEYWORDS,
    PERMITTED_ARDS_NAMES,
    ards_learning_rate,
    ards_schedule,
    ards_schedule_with_warmup,
    get_schedule,
    rational_decay_factor,
    reject_cosine_schedule,
    verify_schedule_purity,
)
from ards.train import (
    SyntheticWindowSampler,
    latest_checkpoint,
    main as train_main,
    run_ards_training,
    save_checkpoint,
)

__all__ = [
    # Schedule
    "DEFAULT_125M_PEAK_LR",
    "DEFAULT_125M_TOTAL_STEPS",
    "DEFAULT_125M_WARMUP_STEPS",
    "DEFAULT_SCALE_FACTOR",
    "FORBIDDEN_COSINE_KEYWORDS",
    "PERMITTED_ARDS_NAMES",
    "ards_learning_rate",
    "ards_schedule",
    "ards_schedule_with_warmup",
    "get_schedule",
    "rational_decay_factor",
    "reject_cosine_schedule",
    "validate_schedule_not_cosine",
    "verify_schedule_purity",
    # Configurations & Presets
    "standard_ards",
    "algebraic_ards",
    "ARDS_PRESETS",
    "get_ards_config",
    "ARDSTrainingConfig",
    "get_training_config",
    # Training Pipeline
    "SyntheticWindowSampler",
    "latest_checkpoint",
    "run_ards_training",
    "save_checkpoint",
    "train_main",
    # Evaluation Pipeline
    "evaluate_ards_run",
    "evaluate_main",
    "evaluate_recall",
    "evaluate_zero_shot",
    "run_ards_evaluation",
    "validate_ards_checkpoint_purity",
]
