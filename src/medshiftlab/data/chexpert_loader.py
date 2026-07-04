"""CheXpert metadata CSV loader for MedShiftLab-CXR.

This module loads CheXpert-style metadata CSV files into validated
CheXpertRecord objects. It does not load raw images or run model inference.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from medshiftlab.data.chexpert import CHEXPERT_PATH_COLUMN, CheXpertRecord, parse_chexpert_record
from medshiftlab.labels.ontology import CXRLabelOntology
from medshiftlab.labels.uncertainty import UncertaintyStrategy


def load_chexpert_metadata_csv(
    csv_path: str | Path,
    ontology: CXRLabelOntology,
    strategy: UncertaintyStrategy | str,
    *,
    soft_value: float = 0.5,
    max_rows: int | None = None,
) -> list[CheXpertRecord]:
    """Load a CheXpert-style metadata CSV into validated records.

    Args:
        csv_path: Path to a CheXpert-style metadata CSV.
        ontology: Validated MedShiftLab-CXR label ontology.
        strategy: CheXpert uncertainty-label handling strategy.
        soft_value: Value used for uncertain labels under U-soft.
        max_rows: Optional row limit for smoke tests or small audits.

    Returns:
        A list of validated CheXpertRecord objects.
    """

    metadata_path = Path(csv_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"CheXpert metadata CSV not found: {metadata_path}")

    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided")

    dataframe = pd.read_csv(metadata_path, nrows=max_rows)

    if CHEXPERT_PATH_COLUMN not in dataframe.columns:
        raise ValueError(f"Missing required CheXpert column: {CHEXPERT_PATH_COLUMN}")

    records: list[CheXpertRecord] = []

    for row in dataframe.to_dict(orient="records"):
        records.append(
            parse_chexpert_record(
                row,
                ontology,
                strategy,
                soft_value=soft_value,
            )
        )

    return records
