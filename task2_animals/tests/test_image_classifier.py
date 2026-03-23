from pathlib import Path

import pandas as pd

from vision.loaders.animal_dataset_loader import AnimalDatasetLoader
from vision.models.image_classifier import AnimalImageClassifier


def test_image_classifier_trains_and_predicts(tmp_path: Path, fake_downloader, tiny_model_factory):
    loader = AnimalDatasetLoader(dataset_downloader=fake_downloader, seed=3)
    dataset_info = loader.load(max_classes=2)
    manifest_path = tmp_path / "image_manifest.csv"
    dataset_info.image_manifest.to_csv(manifest_path, index=False)

    model = AnimalImageClassifier(device="cpu", image_size=32, model_factory=tiny_model_factory)
    result = model.train(
        manifest_path=manifest_path,
        model_dir=tmp_path / "image_model",
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-2,
    )

    test_frame = pd.read_csv(manifest_path)
    sample_image = Path(test_frame[test_frame["split"] == "test"].iloc[0]["image_path"])
    prediction = model.predict(sample_image)

    assert Path(result["model_dir"]).exists()
    assert prediction["label"] in result["class_names"]
