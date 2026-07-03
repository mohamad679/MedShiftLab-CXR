"""Label ontology loading and validation for MedShiftLab-CXR.

This module validates the conservative common-label mapping used to connect
CheXpert internal analysis with VinDr-CXR external validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LabelMapping(BaseModel):
    """Mapping from one MedShiftLab-CXR label to source-dataset labels."""

    model_config = ConfigDict(extra="forbid")

    project_label: str = Field(min_length=1)
    chexpert_label: str = Field(min_length=1)
    vindr_label: str = Field(min_length=1)
    status: str = Field(min_length=1)
    notes: str = Field(min_length=1)

    @field_validator("project_label", "chexpert_label", "vindr_label", "status", "notes")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("label ontology fields must not be empty")
        return value


class CXRLabelOntology(BaseModel):
    """Validated common CXR label ontology."""

    model_config = ConfigDict(extra="forbid")

    project: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: str = Field(min_length=1)
    core_labels: list[LabelMapping] = Field(min_length=1)
    normal_label: LabelMapping

    @model_validator(mode="after")
    def _validate_unique_labels(self) -> CXRLabelOntology:
        project_labels = [label.project_label for label in self.core_labels]
        chexpert_labels = [label.chexpert_label for label in self.core_labels]
        vindr_labels = [label.vindr_label for label in self.core_labels]

        _raise_if_duplicates(project_labels, "project_label")
        _raise_if_duplicates(chexpert_labels, "chexpert_label")
        _raise_if_duplicates(vindr_labels, "vindr_label")

        if self.normal_label.project_label in project_labels:
            raise ValueError("normal_label project_label must not duplicate a core label")

        return self

    @property
    def core_project_labels(self) -> tuple[str, ...]:
        """Return project-level pathology labels in configured order."""

        return tuple(label.project_label for label in self.core_labels)

    @property
    def all_project_labels(self) -> tuple[str, ...]:
        """Return core pathology labels plus the separately analyzed normal label."""

        return (*self.core_project_labels, self.normal_label.project_label)

    def chexpert_to_project(self) -> dict[str, str]:
        """Return mapping from CheXpert source labels to project labels."""

        mapping = {label.chexpert_label: label.project_label for label in self.core_labels}
        mapping[self.normal_label.chexpert_label] = self.normal_label.project_label
        return mapping

    def vindr_to_project(self) -> dict[str, str]:
        """Return mapping from VinDr-CXR source labels to project labels."""

        mapping = {label.vindr_label: label.project_label for label in self.core_labels}
        mapping[self.normal_label.vindr_label] = self.normal_label.project_label
        return mapping


def load_label_ontology(path: str | Path) -> CXRLabelOntology:
    """Load and validate a MedShiftLab-CXR label ontology YAML file."""

    ontology_path = Path(path)
    if not ontology_path.exists():
        raise FileNotFoundError(f"Label ontology file not found: {ontology_path}")

    with ontology_path.open("r", encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file)

    if not isinstance(raw_data, dict):
        raise ValueError("Label ontology YAML must contain a mapping at the top level")

    return CXRLabelOntology.model_validate(raw_data)


def load_default_label_ontology() -> CXRLabelOntology:
    """Load the repository-default CXR common-label ontology."""

    repo_root = Path(__file__).resolve().parents[3]
    return load_label_ontology(repo_root / "configs" / "labels" / "cxr_common_labels.yaml")


def _raise_if_duplicates(values: list[str], field_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate {field_name} values in label ontology: {duplicate_list}")
