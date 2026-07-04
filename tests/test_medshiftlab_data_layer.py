"""Tests for the MedShiftLab-CXR data layer."""

from __future__ import annotations

import math

import pytest

from medshiftlab.data import infer_chexpert_patient_id, parse_chexpert_record
from medshiftlab.labels import (
    UncertaintyStrategy,
    load_default_label_ontology,
    load_label_ontology,
    transform_chexpert_label_mapping,
    transform_chexpert_label_value,
)


def _chexpert_row(image_path: str) -> dict[str, object]:
    return {
        "Path": image_path,
        "Atelectasis": 0,
        "Cardiomegaly": 0,
        "Pleural Effusion": 0,
        "Pneumonia": 0,
        "Pneumothorax": 0,
        "No Finding": 1,
    }


def test_default_label_ontology_loads_core_labels() -> None:
    ontology = load_default_label_ontology()

    assert ontology.project == "MedShiftLab-CXR"
    assert ontology.core_project_labels == (
        "Atelectasis",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
    )
    assert ontology.all_project_labels == (
        "Atelectasis",
        "Cardiomegaly",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
        "No Finding",
    )


def test_label_ontology_rejects_duplicate_project_labels(tmp_path) -> None:
    ontology_path = tmp_path / "bad_labels.yaml"
    ontology_path.write_text(
        """
project: MedShiftLab-CXR
version: 0.1.0
status: test
core_labels:
  - project_label: Atelectasis
    chexpert_label: Atelectasis
    vindr_label: Atelectasis
    status: core
    notes: first
  - project_label: Atelectasis
    chexpert_label: Cardiomegaly
    vindr_label: Cardiomegaly
    status: core
    notes: duplicate project label
normal_label:
  project_label: No Finding
  chexpert_label: No Finding
  vindr_label: No finding
  status: analyzed_separately
  notes: normal label
""",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="Duplicate project_label"):
        load_label_ontology(ontology_path)


@pytest.mark.parametrize(
    ("strategy", "expected_uncertain_value"),
    [
        (UncertaintyStrategy.IGNORE, None),
        (UncertaintyStrategy.ZERO, 0.0),
        (UncertaintyStrategy.ONE, 1.0),
        (UncertaintyStrategy.SOFT, 0.5),
    ],
)
def test_uncertainty_strategy_transforms_uncertain_label(
    strategy: UncertaintyStrategy,
    expected_uncertain_value: float | None,
) -> None:
    assert transform_chexpert_label_value(-1, strategy) == expected_uncertain_value


@pytest.mark.parametrize("missing_value", [None, "", "nan", "NA"])
def test_uncertainty_strategy_keeps_missing_values_missing(missing_value) -> None:
    assert transform_chexpert_label_value(missing_value, "U-one") is None


def test_uncertainty_strategy_handles_nan() -> None:
    assert transform_chexpert_label_value(math.nan, "U-zero") is None


def test_uncertainty_strategy_rejects_invalid_label_value() -> None:
    with pytest.raises(ValueError, match="CheXpert label values"):
        transform_chexpert_label_value(2, "U-zero")


def test_uncertainty_mapping_transforms_multiple_labels() -> None:
    labels = {
        "Atelectasis": -1,
        "Cardiomegaly": 1,
        "Pleural Effusion": 0,
        "Pneumonia": "",
    }

    assert transform_chexpert_label_mapping(labels, "U-soft") == {
        "Atelectasis": 0.5,
        "Cardiomegaly": 1.0,
        "Pleural Effusion": 0.0,
        "Pneumonia": None,
    }


def test_infer_chexpert_patient_id_from_path() -> None:
    image_path = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"

    assert infer_chexpert_patient_id(image_path) == "patient00001"


def test_infer_chexpert_patient_id_returns_none_for_unknown_path() -> None:
    assert infer_chexpert_patient_id("not/a/chexpert/path.jpg") is None


def test_parse_chexpert_record_maps_labels_and_metadata() -> None:
    ontology = load_default_label_ontology()
    row = {
        "Path": "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
        "Sex": "Female",
        "Age": "65",
        "Frontal/Lateral": "Frontal",
        "AP/PA": "PA",
        "Atelectasis": -1,
        "Cardiomegaly": 1,
        "Pleural Effusion": 0,
        "Pneumonia": "",
        "Pneumothorax": None,
        "No Finding": 0,
    }

    record = parse_chexpert_record(row, ontology, "U-soft")

    assert record.dataset_name == "CheXpert"
    assert record.patient_id == "patient00001"
    assert record.sex == "Female"
    assert record.age == 65.0
    assert record.view_position == "Frontal"
    assert record.ap_pa == "PA"
    assert record.labels == {
        "Atelectasis": 0.5,
        "Cardiomegaly": 1.0,
        "Pleural Effusion": 0.0,
        "Pneumonia": None,
        "Pneumothorax": None,
        "No Finding": 0.0,
    }


def test_parse_chexpert_record_rejects_missing_path() -> None:
    ontology = load_default_label_ontology()

    with pytest.raises(ValueError, match="Missing required CheXpert column"):
        parse_chexpert_record({}, ontology, "U-ignore")


def test_load_chexpert_metadata_csv_loads_tiny_csv(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import load_chexpert_metadata_csv
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "chexpert_tiny.csv"

    pd.DataFrame(
        [
            {
                "Path": "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                "Sex": "Female",
                "Age": 65,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "PA",
                "Atelectasis": -1,
                "Cardiomegaly": 1,
                "Pleural Effusion": 0,
                "Pneumonia": "",
                "Pneumothorax": None,
                "No Finding": 0,
            },
            {
                "Path": "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
                "Sex": "Male",
                "Age": 72,
                "Frontal/Lateral": "Frontal",
                "AP/PA": "AP",
                "Atelectasis": 0,
                "Cardiomegaly": -1,
                "Pleural Effusion": 1,
                "Pneumonia": 0,
                "Pneumothorax": "",
                "No Finding": 0,
            },
        ]
    ).to_csv(csv_path, index=False)

    records = load_chexpert_metadata_csv(csv_path, ontology, "U-soft")

    assert len(records) == 2
    assert records[0].patient_id == "patient00001"
    assert records[0].labels["Atelectasis"] == 0.5
    assert records[1].patient_id == "patient00002"
    assert records[1].labels["Cardiomegaly"] == 0.5


def test_load_chexpert_metadata_csv_respects_max_rows(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import load_chexpert_metadata_csv
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "chexpert_tiny.csv"

    pd.DataFrame(
        [
            _chexpert_row(
                "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
            ),
            _chexpert_row(
                "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg"
            ),
        ]
    ).to_csv(csv_path, index=False)

    records = load_chexpert_metadata_csv(csv_path, ontology, "U-ignore", max_rows=1)

    assert len(records) == 1
    assert records[0].patient_id == "patient00001"


def test_iter_chexpert_metadata_csv_preserves_order_across_chunks(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import iter_chexpert_metadata_csv

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "chexpert_tiny.csv"
    paths = [
        f"CheXpert-v1.0-small/train/patient0000{index}/study1/view1_frontal.jpg"
        for index in range(1, 4)
    ]
    pd.DataFrame([_chexpert_row(path) for path in paths]).to_csv(
        csv_path, index=False
    )

    records = list(
        iter_chexpert_metadata_csv(
            csv_path,
            ontology,
            "U-ignore",
            chunksize=1,
        )
    )

    assert [record.image_path for record in records] == paths


def test_iter_chexpert_metadata_csv_respects_max_rows(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import iter_chexpert_metadata_csv

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "chexpert_tiny.csv"
    rows = [
        _chexpert_row(
            f"CheXpert-v1.0-small/train/patient0000{index}/study1/view1_frontal.jpg"
        )
        for index in range(1, 5)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    records = list(
        iter_chexpert_metadata_csv(
            csv_path,
            ontology,
            "U-ignore",
            max_rows=3,
            chunksize=2,
        )
    )

    assert len(records) == 3
    assert records[-1].patient_id == "patient00003"


def test_iter_chexpert_metadata_csv_rejects_nonpositive_chunksize(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import iter_chexpert_metadata_csv

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "chexpert_tiny.csv"
    pd.DataFrame([_chexpert_row("local/view.jpg")]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="chunksize must be positive"):
        list(
            iter_chexpert_metadata_csv(
                csv_path,
                ontology,
                "U-ignore",
                chunksize=0,
            )
        )


def test_iter_chexpert_metadata_csv_validates_columns_before_yielding(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import iter_chexpert_metadata_csv

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "bad_chexpert.csv"
    row = _chexpert_row("local/view.jpg")
    del row["Pneumonia"]
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    iterator = iter_chexpert_metadata_csv(
        csv_path,
        ontology,
        "U-ignore",
        chunksize=1,
    )
    with pytest.raises(
        ValueError,
        match="Missing required CheXpert metadata columns: Pneumonia",
    ):
        next(iterator)


def test_load_chexpert_metadata_csv_rejects_missing_file(tmp_path) -> None:
    from medshiftlab.data import load_chexpert_metadata_csv
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()

    with pytest.raises(FileNotFoundError, match="CheXpert metadata CSV not found"):
        load_chexpert_metadata_csv(tmp_path / "missing.csv", ontology, "U-ignore")


def test_load_chexpert_metadata_csv_rejects_missing_path_column(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import load_chexpert_metadata_csv
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "bad_chexpert.csv"

    pd.DataFrame([{"Atelectasis": 1}]).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match="Missing required CheXpert metadata columns: Path",
    ):
        load_chexpert_metadata_csv(csv_path, ontology, "U-ignore")


def test_load_chexpert_metadata_csv_rejects_missing_label_column(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import load_chexpert_metadata_csv

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "bad_chexpert.csv"
    row = _chexpert_row(
        "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    )
    del row["Pneumonia"]
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    with pytest.raises(
        ValueError,
        match="Missing required CheXpert metadata columns: Pneumonia",
    ):
        load_chexpert_metadata_csv(csv_path, ontology, "U-ignore")


def test_summarize_chexpert_records_counts_patients_and_labels(tmp_path) -> None:
    import pandas as pd

    from medshiftlab.data import load_chexpert_metadata_csv, summarize_chexpert_records
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()
    csv_path = tmp_path / "chexpert_tiny.csv"

    pd.DataFrame(
        [
            {
                "Path": "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                "Atelectasis": -1,
                "Cardiomegaly": 1,
                "Pleural Effusion": 0,
                "Pneumonia": "",
                "Pneumothorax": None,
                "No Finding": 0,
            },
            {
                "Path": "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
                "Atelectasis": 0,
                "Cardiomegaly": -1,
                "Pleural Effusion": 1,
                "Pneumonia": 0,
                "Pneumothorax": "",
                "No Finding": 0,
            },
        ]
    ).to_csv(csv_path, index=False)

    records = load_chexpert_metadata_csv(csv_path, ontology, "U-soft")
    summary = summarize_chexpert_records(records)

    assert summary.dataset_name == "CheXpert"
    assert summary.n_records == 2
    assert summary.n_patients == 2
    assert summary.n_records_without_patient_id == 0
    assert summary.labels["Atelectasis"].available_count == 2
    assert summary.labels["Atelectasis"].soft_count == 1
    assert summary.labels["Atelectasis"].mean_target == 0.25
    assert summary.labels["Cardiomegaly"].positive_count == 1
    assert summary.labels["Cardiomegaly"].soft_count == 1
    assert summary.labels["Cardiomegaly"].mean_target == 0.75


def test_summarize_chexpert_records_rejects_empty_collection() -> None:
    from medshiftlab.data import summarize_chexpert_records

    with pytest.raises(ValueError, match="Cannot summarize an empty record collection"):
        summarize_chexpert_records([])


def test_parse_vindr_cxr_record_maps_binary_labels() -> None:
    from medshiftlab.data import parse_vindr_cxr_record
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()
    row = {
        "image_id": "00000001",
        "image_path": "vindr-cxr/train/00000001.jpg",
        "split": "test",
        "view_position": "PA",
        "Atelectasis": 1,
        "Cardiomegaly": 0,
        "Pleural effusion": 1,
        "Pneumonia": "",
        "Pneumothorax": None,
        "No finding": 0,
    }

    record = parse_vindr_cxr_record(row, ontology)

    assert record.dataset_name == "VinDr-CXR"
    assert record.image_id == "00000001"
    assert record.image_path == "vindr-cxr/train/00000001.jpg"
    assert record.split == "test"
    assert record.view_position == "PA"
    assert record.labels == {
        "Atelectasis": 1.0,
        "Cardiomegaly": 0.0,
        "Pleural Effusion": 1.0,
        "Pneumonia": None,
        "Pneumothorax": None,
        "No Finding": 0.0,
    }


def test_parse_vindr_cxr_record_rejects_missing_image_id() -> None:
    from medshiftlab.data import parse_vindr_cxr_record
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()

    with pytest.raises(ValueError, match="Missing required VinDr-CXR image ID column"):
        parse_vindr_cxr_record({}, ontology)


def test_parse_vindr_cxr_record_rejects_non_binary_label() -> None:
    from medshiftlab.data import parse_vindr_cxr_record
    from medshiftlab.labels import load_default_label_ontology

    ontology = load_default_label_ontology()
    row = {
        "image_id": "00000001",
        "Atelectasis": 2,
        "Cardiomegaly": 0,
        "Pleural effusion": 0,
        "Pneumonia": 0,
        "Pneumothorax": 0,
        "No finding": 0,
    }

    with pytest.raises(ValueError, match="VinDr-CXR image-level labels must be binary"):
        parse_vindr_cxr_record(row, ontology)


def test_parse_vindr_cxr_record_rejects_missing_label_column() -> None:
    from medshiftlab.data import parse_vindr_cxr_record

    ontology = load_default_label_ontology()
    row = {
        "image_id": "00000001",
        "Atelectasis": 0,
        "Cardiomegaly": 0,
        "Pleural effusion": 0,
        "Pneumothorax": 0,
        "No finding": 1,
    }

    with pytest.raises(
        ValueError,
        match="Missing required VinDr-CXR label columns: Pneumonia",
    ):
        parse_vindr_cxr_record(row, ontology)


def test_validate_patient_disjoint_splits_rejects_cross_split_patient() -> None:
    from medshiftlab.data import validate_patient_disjoint_splits

    ontology = load_default_label_ontology()
    record = parse_chexpert_record(
        _chexpert_row(
            "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        ),
        ontology,
        "U-ignore",
    )

    with pytest.raises(ValueError, match="appears in multiple splits"):
        validate_patient_disjoint_splits(
            {"train": [record], "validation": [record]}
        )


def test_validate_patient_disjoint_splits_allows_repeated_patient_within_split() -> None:
    from medshiftlab.data import validate_patient_disjoint_splits

    ontology = load_default_label_ontology()
    first_record = parse_chexpert_record(
        _chexpert_row(
            "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        ),
        ontology,
        "U-ignore",
    )
    second_record = parse_chexpert_record(
        _chexpert_row(
            "CheXpert-v1.0-small/train/patient00001/study2/view1_frontal.jpg"
        ),
        ontology,
        "U-ignore",
    )

    validate_patient_disjoint_splits({"train": [first_record, second_record]})


def test_validate_patient_disjoint_splits_ignores_missing_patient_ids() -> None:
    from medshiftlab.data import validate_patient_disjoint_splits

    ontology = load_default_label_ontology()
    record = parse_chexpert_record(
        _chexpert_row("local/images/view1.jpg"),
        ontology,
        "U-ignore",
    )

    assert record.patient_id is None
    validate_patient_disjoint_splits({"train": [record], "validation": [record]})


def test_validate_patient_disjoint_splits_accepts_disjoint_named_splits() -> None:
    from medshiftlab.data import validate_patient_disjoint_splits

    ontology = load_default_label_ontology()
    records = [
        parse_chexpert_record(
            _chexpert_row(
                f"CheXpert-v1.0-small/train/patient0000{index}/study1/view1_frontal.jpg"
            ),
            ontology,
            "U-ignore",
        )
        for index in range(1, 4)
    ]

    validate_patient_disjoint_splits(
        {"train": [records[0]], "validation": [records[1]], "test": [records[2]]}
    )


def test_validate_patient_disjoint_splits_rejects_blank_split_name() -> None:
    from medshiftlab.data import validate_patient_disjoint_splits

    with pytest.raises(ValueError, match="split names must not be blank"):
        validate_patient_disjoint_splits({" ": []})


def test_validate_patient_disjoint_splits_rejects_blank_patient_id() -> None:
    from medshiftlab.data import validate_patient_disjoint_splits

    ontology = load_default_label_ontology()
    record = parse_chexpert_record(
        _chexpert_row("local/images/view1.jpg"),
        ontology,
        "U-ignore",
    ).model_copy(update={"patient_id": " "})

    with pytest.raises(ValueError, match="patient IDs must not be blank"):
        validate_patient_disjoint_splits({"train": [record]})
