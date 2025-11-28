"""
Analysis package initializer.

This module exposes the main analysis functions so they can be imported
directly from the package, for example:

    from Src.analysis import generate_integration_figure

The __all__ list defines the public API of the analysis package,
specifying which functions are intended for external use.
"""

from .analysis import (
    generate_integration_figure,
    save_segment_summary_png,
    define_global_classes,
    save_class_summary_png,
    extract_acc_features,
    save_features_png,
    train_random_forest_acc,
)

__all__ = [
    "generate_integration_figure",
    "save_segment_summary_png",
    "define_global_classes",
    "save_class_summary_png",
    "extract_acc_features",
    "save_features_png",
    "train_random_forest_acc",
]
