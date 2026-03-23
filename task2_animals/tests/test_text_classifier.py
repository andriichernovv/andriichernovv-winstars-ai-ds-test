from pathlib import Path

from data.dataset_service import DatasetService
from text.loaders.text_dataset_loader import TextDatasetLoader
from text.models.text_classifier import AnimalTextClassifier
from vision.loaders.animal_dataset_loader import AnimalDatasetLoader


def test_text_classifier_trains_and_extracts_animal(tmp_path: Path, fake_downloader):
    image_loader = AnimalDatasetLoader(dataset_downloader=fake_downloader, seed=5)
    dataset_service = DatasetService(dataset_loader=image_loader, data_dir=tmp_path / "data", seed=5)
    prepared = dataset_service.prepare_datasets(max_classes=2, text_samples_per_class=6)

    model = AnimalTextClassifier(
        dataset_loader=TextDatasetLoader(),
        device="cpu",
        max_length=24,
    )
    result = model.train(
        dataset_dir=prepared["text_dataset_dir"],
        model_dir=tmp_path / "text_model",
        num_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )

    animal = model.extract_animal("I think this is a cat.")

    assert Path(result["model_dir"]).exists()
    assert animal == "cat"
