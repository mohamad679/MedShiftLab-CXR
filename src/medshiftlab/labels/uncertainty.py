"""CheXpert uncertainty-label handling for MedShiftLab-CXR.

The functions in this module transform CheXpert-style label values into
research targets under predefined uncertainty strategies.
"""

from __future__ import annotations

from enum import Enum
from math import isnan
from typing import Any, Mapping


class UncertaintyStrategy(str, Enum):
    """Supported CheXpert uncertainty-label strategies."""

    IGNORE = "U-ignore"
    ZERO = "U-zero"
    ONE = "U-one"
    SOFT = "U-soft"


def transform_chexpert_label_value(
    value: Any,
    strategy: UncertaintyStrategy | str,
    *,
    soft_value: float = 0.5,
) -> float | None:
    """Transform one CheXpert-style label value under an uncertainty strategy.

    CheXpert-style values:

    - 1: positive
    - 0: negative
    - -1: uncertain
    - missing/blank/NaN: missing label

    Returns:
        1.0, 0.0, soft_value, or None.
        None means the label should be excluded from supervised metric
        computation for that sample and label.
    """

    parsed_strategy = parse_uncertainty_strategy(strategy)

    if not 0.0 <= soft_value <= 1.0:
        raise ValueError("soft_value must be between 0.0 and 1.0")

    normalized_value = _normalize_label_value(value)

    if normalized_value is None:
        return None

    if normalized_value == 1.0:
        return 1.0

    if normalized_value == 0.0:
        return 0.0

    if normalized_value == -1.0:
        if parsed_strategy is UncertaintyStrategy.IGNORE:
            return None
        if parsed_strategy is UncertaintyStrategy.ZERO:
            return 0.0
        if parsed_strategy is UncertaintyStrategy.ONE:
            return 1.0
        if parsed_strategy is UncertaintyStrategy.SOFT:
            return float(soft_value)

    raise ValueError(
        "CheXpert label values must be one of 1, 0, -1, missing, blank, or NaN. "
        f"Received: {value!r}"
    )


def transform_chexpert_label_mapping(
    labels: Mapping[str, Any],
    strategy: UncertaintyStrategy | str,
    *,
    soft_value: float = 0.5,
) -> dict[str, float | None]:
    """Transform a mapping of label names to CheXpert-style values."""

    return {
        label_name: transform_chexpert_label_value(
            value,
            strategy,
            soft_value=soft_value,
        )
        for label_name, value in labels.items()
    }


def parse_uncertainty_strategy(strategy: UncertaintyStrategy | str) -> UncertaintyStrategy:
    """Parse and validate an uncertainty strategy."""

    if isinstance(strategy, UncertaintyStrategy):
        return strategy

    try:
        return UncertaintyStrategy(strategy)
    except ValueError as exc:
        valid = ", ".join(item.value for item in UncertaintyStrategy)
        raise ValueError(f"Unknown uncertainty strategy {strategy!r}. Valid values: {valid}") from exc


def _normalize_label_value(value: Any) -> float | None:
    """Normalize raw CheXpert-style values into 1.0, 0.0, -1.0, or None."""

    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"nan", "none", "null", "na"}:
            return None
        value = stripped

    if isinstance(value, float) and isnan(value):
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CheXpert label value: {value!r}") from exc

    if numeric_value in {1.0, 0.0, -1.0}:
        return numeric_value

    raise ValueError(
        "CheXpert label values must be one of 1, 0, -1, missing, blank, or NaN. "
        f"Received: {value!r}"
    )
