"""Exact explanation utilities for the deployed multinomial linear model."""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Rank features by mean absolute coefficient across all classes."""
    importance = np.abs(model.coef_).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_coefficient": importance})
        .sort_values("mean_abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def local_contributions(model, row: pd.DataFrame, predicted_class: int) -> pd.DataFrame:
    """Return exact class-score contributions for a single transformed row."""
    class_position = list(model.classes_).index(predicted_class)
    values = row.iloc[0].to_numpy(dtype=float)
    contributions = values * model.coef_[class_position]
    return (
        pd.DataFrame({"feature": row.columns, "value": values, "contribution": contributions})
        .assign(abs_contribution=lambda frame: frame["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )
