"""Focused tests for Phase 5 baseline inference integration."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml
from PIL import Image

from medshiftlab.data import load_inference_manifest_csv, load_local_data_config
from medshiftlab.models import TorchXRayVisionAdapter, TorchXRayVisionAdapterConfig
from medshiftlab.reporting import read_prediction_batch_json, read_prediction_records_csv


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeTensor:
    def __init__(self, array: object) -> None:
        import numpy as np

        self.array = np.asarray(array, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    def unsqueeze(self, axis: int) -> _FakeTensor:
        import numpy as np

        return _FakeTensor(np.expand_dims(self.array, axis))

    def permute(self, *axes: int) -> _FakeTensor:
        import numpy as np

        return _FakeTensor(np.transpose(self.array, axes))

    def to(self, _device: str) -> _FakeTensor:
        return self

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> list[object]:
        return self.array.tolist()


class _FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def as_tensor(array: object) -> _FakeTensor:
        return _FakeTensor(array)

    @staticmethod
    def stack(tensors: list[_FakeTensor]) -> _FakeTensor:
        import numpy as np

        return _FakeTensor(np.stack([tensor.array for tensor in tensors], axis=0))

    @staticmethod
    def inference_mode() -> nullcontext[None]:
        return nullcontext()


class _FakeModel:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.eval_called = False
        self._offset = 0

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def __call__(self, batch: _FakeTensor) -> _FakeTensor:
        batch_size = batch.array.shape[0]
        start = self._offset
        end = start + batch_size
        self._offset = end
        return _FakeTensor(self.rows[start:end])


def _environment(*extra_pythonpath: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in extra_pythonpath), "src", "."]
    )
    return environment


def _write_local_data_config(tmp_path: Path, image_dir: Path) -> Path:
    config_path = tmp_path / "local_paths.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "datasets": {
                    "chexpert": {
                        "root_path": str(tmp_path / "dataset_root"),
                        "metadata_path": str(tmp_path / "manifest.csv"),
                        "image_directory": str(image_dir),
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_manifest(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_id", "dataset_name", "image_path", "patient_id", "study_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_synthetic_image(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), value).save(path)


def _write_fake_runtime_modules(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "torch.py").write_text(
        """
import numpy as np
from contextlib import contextmanager

class _Tensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
    @property
    def shape(self):
        return self.array.shape
    def unsqueeze(self, axis):
        return _Tensor(np.expand_dims(self.array, axis))
    def permute(self, *axes):
        return _Tensor(np.transpose(self.array, axes))
    def to(self, _device):
        return self
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.array.tolist()

def as_tensor(array):
    return _Tensor(array)

def stack(tensors):
    return _Tensor(np.stack([tensor.array for tensor in tensors], axis=0))

@contextmanager
def inference_mode():
    yield

class cuda:
    @staticmethod
    def is_available():
        return False
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (directory / "torchxrayvision.py").write_text(
        """
import numpy as np
import torch

class _Model:
    pathologies = ["Effusion", "Cardiomegaly", "Pneumonia", "Pneumothorax", "Atelectasis"]
    def to(self, _device):
        return self
    def eval(self):
        return self
    def __call__(self, batch):
        rows = np.tile(np.array([0.3, -0.7, 0.4, -0.2, 1.5], dtype=np.float32), (batch.array.shape[0], 1))
        return torch.as_tensor(rows)

class _Models:
    def DenseNet(self, weights):
        return _Model()

models = _Models()
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_inference_manifest_csv_fills_dataset_name_and_rejects_absolute_paths(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,image_path,patient_id,study_id\nimg001,train/a.png,pat1,study1\n",
        encoding="utf-8",
    )

    records = load_inference_manifest_csv(manifest, dataset_name="chexpert", limit=1)

    assert records[0].sample_id == "img001"
    assert records[0].dataset_name == "chexpert"
    assert records[0].image_path == "train/a.png"

    bad_manifest = tmp_path / "bad_manifest.csv"
    bad_manifest.write_text(
        "sample_id,dataset_name,image_path\nimg001,chexpert,/absolute/path.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative"):
        load_inference_manifest_csv(bad_manifest)


def test_example_manifest_contains_no_private_absolute_paths() -> None:
    example_manifest = REPO_ROOT / "configs" / "data" / "example_inference_manifest.csv"
    payload = example_manifest.read_text(encoding="utf-8")

    assert "sample_id,dataset_name,image_path,patient_id,study_id" in payload
    assert "/Users/mohsenshamsijazeb" not in payload
    assert ":\\" not in payload


def test_adapter_predict_records_returns_standardized_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "images"
    _write_synthetic_image(image_dir / "train" / "img001.png", value=64)
    _write_synthetic_image(image_dir / "train" / "img002.png", value=128)
    local_data_config = load_local_data_config(_write_local_data_config(tmp_path, image_dir))

    monkeypatch.setattr(
        "medshiftlab.models.torchxrayvision_adapter._load_torch_dependency",
        lambda: _FakeTorch,
    )
    adapter = TorchXRayVisionAdapter(
        config=TorchXRayVisionAdapterConfig(
            model_name="torchxrayvision-densenet121",
            model_version="torchxrayvision:test-weights",
            labels=("Atelectasis", "Cardiomegaly"),
            output_indices={"Atelectasis": 0, "Cardiomegaly": 1},
            batch_size=1,
            output_activation="sigmoid",
        ),
        model=_FakeModel([[0.0, 2.0], [-2.0, 1.0]]),
        local_data_config=local_data_config,
    )

    batch = adapter.predict_records(
        [
            {
                "sample_id": "img001",
                "dataset_name": "chexpert",
                "image_path": "train/img001.png",
                "patient_id": "pat1",
                "study_id": "study1",
            },
            {
                "sample_id": "img002",
                "dataset_name": "chexpert",
                "image_path": "train/img002.png",
            },
        ]
    )

    assert batch.model_name == "torchxrayvision-densenet121"
    assert batch.model_version == "torchxrayvision:test-weights"
    assert batch.adapter_name == "torchxrayvision-adapter"
    assert batch.preprocessing_version == "phase5-baseline-inference-v1"
    assert batch.preprocessing_config["image_loader"] == "phase3-package-loader"
    assert batch.preprocessing_config["image_preprocessing"]["normalization"] == "minus_one_one"
    assert batch.run_metadata["n_records"] == 2
    assert batch.records[0].sample_id == "img001"
    assert batch.records[0].patient_id == "pat1"
    assert batch.records[0].study_id == "study1"
    assert batch.records[0].logits == (0.0, 2.0)
    assert batch.records[0].probabilities == pytest.approx((0.5, 0.880797))
    assert batch.records[1].probabilities == pytest.approx((0.119203, 0.731059))


def test_adapter_predict_records_fails_clearly_when_torch_dependency_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "images"
    _write_synthetic_image(image_dir / "train" / "img001.png", value=32)
    local_data_config = load_local_data_config(_write_local_data_config(tmp_path, image_dir))

    def _raise_import_error() -> object:
        raise ImportError("torch not available")

    monkeypatch.setattr(
        "medshiftlab.models.torchxrayvision_adapter._load_torch_dependency",
        _raise_import_error,
    )
    adapter = TorchXRayVisionAdapter(
        config=TorchXRayVisionAdapterConfig(
            model_name="torchxrayvision-densenet121",
            model_version="torchxrayvision:test-weights",
            labels=("Atelectasis",),
            output_indices={"Atelectasis": 0},
        ),
        model=_FakeModel([[0.0]]),
        local_data_config=local_data_config,
    )

    with pytest.raises(ImportError, match="torch not available"):
        adapter.predict_records(
            [
                {
                    "sample_id": "img001",
                    "dataset_name": "chexpert",
                    "image_path": "train/img001.png",
                }
            ]
        )


def test_script_enforces_safe_small_limit_behavior(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_synthetic_image(image_dir / "train" / "img001.png", value=16)
    config_path = _write_local_data_config(tmp_path, image_dir)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": f"img{index:03d}",
                "dataset_name": "chexpert",
                "image_path": "train/img001.png",
                "patient_id": "",
                "study_id": "",
            }
            for index in range(65)
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline_inference.py",
            "--config",
            str(config_path),
            "--dataset",
            "chexpert",
            "--manifest-csv",
            str(manifest_path),
            "--limit",
            "65",
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "safe manual subset threshold" in result.stderr


def test_script_requires_explicit_model_init_flag(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    _write_synthetic_image(image_dir / "train" / "img001.png", value=16)
    config_path = _write_local_data_config(tmp_path, image_dir)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "img001",
                "dataset_name": "chexpert",
                "image_path": "train/img001.png",
                "patient_id": "",
                "study_id": "",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline_inference.py",
            "--config",
            str(config_path),
            "--dataset",
            "chexpert",
            "--manifest-csv",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Model initialization is disabled by default" in result.stderr


def test_script_runs_with_fake_optional_dependencies_and_writes_standardized_outputs(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "fake_modules"
    _write_fake_runtime_modules(fake_modules)
    image_dir = tmp_path / "images"
    _write_synthetic_image(image_dir / "train" / "img001.png", value=32)
    _write_synthetic_image(image_dir / "train" / "img002.png", value=96)
    config_path = _write_local_data_config(tmp_path, image_dir)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "img001",
                "dataset_name": "chexpert",
                "image_path": "train/img001.png",
                "patient_id": "pat1",
                "study_id": "study1",
            },
            {
                "sample_id": "img002",
                "dataset_name": "chexpert",
                "image_path": "train/img002.png",
                "patient_id": "",
                "study_id": "",
            },
        ],
    )
    output_json = tmp_path / "predictions.json"
    output_csv = tmp_path / "predictions.csv"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline_inference.py",
            "--config",
            str(config_path),
            "--dataset",
            "chexpert",
            "--manifest-csv",
            str(manifest_path),
            "--allow-model-init",
            "--limit",
            "2",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=REPO_ROOT,
        env=_environment(fake_modules),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["n_records"] == 2
    assert summary["manual_only"] is True
    assert output_json.is_file()
    assert output_csv.is_file()

    batch_from_json = read_prediction_batch_json(output_json)
    batch_from_csv = read_prediction_records_csv(output_csv)
    assert batch_from_json.model_name == "torchxrayvision-densenet121"
    assert batch_from_json.model_version == "torchxrayvision:densenet121-res224-all"
    assert batch_from_json.records[0].image_path == "train/img001.png"
    assert batch_from_json.preprocessing_config["image_loader"] == "phase3-package-loader"
    assert batch_from_json.run_metadata["n_records"] == 2
    assert [record.sample_id for record in batch_from_csv.records] == ["img001", "img002"]
    assert str(image_dir) not in output_json.read_text(encoding="utf-8")
    assert str(image_dir) not in output_csv.read_text(encoding="utf-8")


def test_script_reports_missing_optional_torchxrayvision_dependency(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocked"
    blocker.mkdir()
    (blocker / "torchxrayvision.py").write_text(
        "raise ImportError('blocked for test')\n",
        encoding="utf-8",
    )
    image_dir = tmp_path / "images"
    _write_synthetic_image(image_dir / "train" / "img001.png", value=16)
    config_path = _write_local_data_config(tmp_path, image_dir)
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "sample_id": "img001",
                "dataset_name": "chexpert",
                "image_path": "train/img001.png",
                "patient_id": "",
                "study_id": "",
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_baseline_inference.py",
            "--config",
            str(config_path),
            "--dataset",
            "chexpert",
            "--manifest-csv",
            str(manifest_path),
            "--allow-model-init",
        ],
        cwd=REPO_ROOT,
        env=_environment(blocker),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "optional dependencies" in result.stderr
