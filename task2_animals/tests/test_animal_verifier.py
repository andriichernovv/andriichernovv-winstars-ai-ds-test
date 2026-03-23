from pathlib import Path

import pandas as pd

from data.dataset_service import DatasetService
from pipeline.animal_verifier import AnimalVerifier
from text.models.text_classifier import AnimalTextClassifier
from vision.loaders.animal_dataset_loader import AnimalDatasetLoader
from vision.models.image_classifier import AnimalImageClassifier


def test_animal_verifier_fit_and_verify(tmp_path, fake_downloader, tiny_model_factory):
    loader = AnimalDatasetLoader(dataset_downloader=fake_downloader, seed=11)
    verifier = AnimalVerifier(
        dataset_service=DatasetService(dataset_loader=loader, data_dir=tmp_path / "data", seed=11),
        image_classifier=AnimalImageClassifier(device="cpu", image_size=32, model_factory=tiny_model_factory),
        text_classifier=AnimalTextClassifier(device="cpu", max_length=24),
    )

    result = verifier.fit(
        max_classes=2,
        text_samples_per_class=6,
        image_model_dir=tmp_path / "image_model",
        image_batch_size=2,
        image_num_epochs=1,
        image_learning_rate=1e-2,
        text_model_dir=tmp_path / "text_model",
        text_batch_size=4,
        text_num_epochs=1,
        text_learning_rate=1e-3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )

    image_manifest = pd.read_csv(result["prepared"]["image_manifest_path"])
    sample_row = image_manifest[image_manifest["split"] == "test"].iloc[0]
    details = verifier.verify_details(
        text=f"I think this is a {sample_row['label']}.",
        image_path=sample_row["image_path"],
    )

    assert details["text_animal"] == sample_row["label"]
    assert details["predicted_animal"] in ["cat", "dog"]
    assert isinstance(details["is_match"], bool)


def test_animal_verifier_supports_uploaded_image(tmp_path, fake_downloader, tiny_model_factory):
    loader = AnimalDatasetLoader(dataset_downloader=fake_downloader, seed=13)
    verifier = AnimalVerifier(
        dataset_service=DatasetService(dataset_loader=loader, data_dir=tmp_path / "data", seed=13),
        image_classifier=AnimalImageClassifier(device="cpu", image_size=32, model_factory=tiny_model_factory),
        text_classifier=AnimalTextClassifier(device="cpu", max_length=24),
        data_dir=tmp_path / "data",
    )

    result = verifier.fit(
        max_classes=2,
        text_samples_per_class=6,
        image_model_dir=tmp_path / "image_model",
        image_batch_size=2,
        image_num_epochs=1,
        image_learning_rate=1e-2,
        text_model_dir=tmp_path / "text_model",
        text_batch_size=4,
        text_num_epochs=1,
        text_learning_rate=1e-3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )

    image_manifest = pd.read_csv(result["prepared"]["image_manifest_path"])
    sample_row = image_manifest[image_manifest["split"] == "test"].iloc[0]
    image_bytes = Path(sample_row["image_path"]).read_bytes()

    details = verifier.verify_uploaded_image(
        text=f"I think this is a {sample_row['label']}.",
        image_bytes=image_bytes,
        filename=Path(sample_row["image_path"]).name,
    )

    assert details["text_animal"] == sample_row["label"]
    assert "image_result" in details
