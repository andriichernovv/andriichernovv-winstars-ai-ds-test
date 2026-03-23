import json
from pathlib import Path

import pandas as pd

from data.dataset_service import DatasetService
from vision.loaders.animal_dataset_loader import AnimalDatasetLoader


def test_prepare_datasets_writes_files(tmp_path: Path, fake_downloader):
    loader = AnimalDatasetLoader(dataset_downloader=fake_downloader, seed=7)
    service = DatasetService(dataset_loader=loader, data_dir=tmp_path / "data", seed=7)

    result = service.prepare_datasets(max_classes=2, text_samples_per_class=6)

    manifest = pd.read_csv(result["image_manifest_path"])
    text_metadata = json.loads(
        (Path(result["text_dataset_dir"]) / "metadata.json").read_text(encoding="utf-8")
    )

    assert set(result["class_names"]) == {"cat", "dog"}
    assert not manifest.empty
    assert (Path(result["text_dataset_dir"]) / "train.json").exists()
    assert text_metadata["class_names"] == ["cat", "dog"]


def test_loader_handles_nested_animals_folder(tmp_path: Path):
    nested_root = tmp_path / "download_cache" / "animals" / "animals"
    nested_root.mkdir(parents=True, exist_ok=True)

    for class_name in ("cat", "dog"):
        class_dir = nested_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / f"{class_name}_0.png").write_bytes(b"fake")

    loader = AnimalDatasetLoader(dataset_downloader=lambda handle: str(tmp_path / "download_cache"))
    image_root = loader._find_image_root(tmp_path / "download_cache")

    assert image_root == nested_root


def test_loader_uses_local_kagglehub_cache_before_downloading(tmp_path: Path, monkeypatch):
    cached_root = (
        tmp_path
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "iamsouravbanerjee"
        / "animal-image-dataset-90-different-animals"
        / "versions"
        / "5"
        / "animals"
        / "animals"
    )
    cached_root.mkdir(parents=True, exist_ok=True)

    for class_name in ("cat", "dog"):
        class_dir = cached_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / f"{class_name}_0.png").write_bytes(b"fake")
        (class_dir / f"{class_name}_1.png").write_bytes(b"fake")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "vision.loaders.animal_dataset_loader.kagglehub.dataset_download",
        lambda handle: (_ for _ in ()).throw(AssertionError("network should not be used")),
    )

    loader = AnimalDatasetLoader(seed=5)
    dataset_info = loader.load(max_classes=2)

    assert dataset_info.image_root == cached_root
    assert dataset_info.class_names == ["cat", "dog"]
