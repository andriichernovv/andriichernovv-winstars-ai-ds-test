from pathlib import Path
import uuid

from data.dataset_service import DatasetService
from vision.loaders.animal_dataset_loader import AnimalDatasetLoader
from vision.models.image_classifier import AnimalImageClassifier
from text.models.text_classifier import AnimalTextClassifier


class AnimalVerifier:
    def __init__(
        self,
        dataset_service: DatasetService | None = None,
        image_classifier: AnimalImageClassifier | None = None,
        text_classifier: AnimalTextClassifier | None = None,
        data_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_dir = Path(data_dir) if data_dir is not None else self.project_root / "data"
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir is not None else self.project_root / "artifacts"
        )
        self.dataset_service = dataset_service or DatasetService(
            dataset_loader=AnimalDatasetLoader(),
            data_dir=self.data_dir,
        )
        self.image_classifier = image_classifier or AnimalImageClassifier()
        self.text_classifier = text_classifier or AnimalTextClassifier()
        self.prepared_paths: dict | None = None

    def prepare_datasets(
        self,
        max_classes: int | None = None,
        text_samples_per_class: int = 32,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict:
        self.prepared_paths = self.dataset_service.prepare_datasets(
            max_classes=max_classes,
            text_samples_per_class=text_samples_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        return self.prepared_paths

    def train_image(
        self,
        image_manifest_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        batch_size: int = 32,
        num_epochs: int = 8,
        learning_rate: float = 3e-4,
        num_workers: int = 0,
    ) -> dict:
        manifest_path = image_manifest_path or self._require_prepared_path("image_manifest_path")
        resolved_model_dir = Path(model_dir) if model_dir is not None else self.artifacts_dir / "image_model"
        return self.image_classifier.train(
            manifest_path=manifest_path,
            model_dir=resolved_model_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            num_workers=num_workers,
        )

    def train_text(
        self,
        text_dataset_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 5e-4,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 128,
    ) -> dict:
        dataset_dir = text_dataset_dir or self._require_prepared_path("text_dataset_dir")
        resolved_model_dir = Path(model_dir) if model_dir is not None else self.artifacts_dir / "text_model"
        return self.text_classifier.train(
            dataset_dir=dataset_dir,
            model_dir=resolved_model_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )

    def fit(
        self,
        max_classes: int | None = None,
        text_samples_per_class: int = 32,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        image_model_dir: str | Path | None = None,
        image_batch_size: int = 32,
        image_num_epochs: int = 8,
        image_learning_rate: float = 3e-4,
        image_num_workers: int = 0,
        text_model_dir: str | Path | None = None,
        text_batch_size: int = 16,
        text_num_epochs: int = 5,
        text_learning_rate: float = 5e-4,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 128,
    ) -> dict:
        prepared = self.prepare_datasets(
            max_classes=max_classes,
            text_samples_per_class=text_samples_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        image_training = self.train_image(
            image_manifest_path=prepared["image_manifest_path"],
            model_dir=image_model_dir,
            batch_size=image_batch_size,
            num_epochs=image_num_epochs,
            learning_rate=image_learning_rate,
            num_workers=image_num_workers,
        )
        text_training = self.train_text(
            text_dataset_dir=prepared["text_dataset_dir"],
            model_dir=text_model_dir,
            batch_size=text_batch_size,
            num_epochs=text_num_epochs,
            learning_rate=text_learning_rate,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )
        return {
            "prepared": prepared,
            "image_training": image_training,
            "text_training": text_training,
        }

    def load(
        self,
        image_model_dir: str | Path | None = None,
        text_model_dir: str | Path | None = None,
    ) -> None:
        resolved_image_model_dir = (
            Path(image_model_dir) if image_model_dir is not None else self.artifacts_dir / "image_model"
        )
        resolved_text_model_dir = (
            Path(text_model_dir) if text_model_dir is not None else self.artifacts_dir / "text_model"
        )
        self.image_classifier.load(resolved_image_model_dir)
        self.text_classifier.load(resolved_text_model_dir)

    def predict_image(self, image_path: str | Path, top_k: int = 5) -> dict:
        return self.image_classifier.predict(image_path=image_path, top_k=top_k)

    def extract_animal(self, text: str) -> str | None:
        return self.text_classifier.extract_animal(text)

    def verify(self, text: str, image_path: str | Path, top_k: int = 5) -> bool:
        return bool(self.verify_details(text=text, image_path=image_path, top_k=top_k)["is_match"])

    def verify_details(self, text: str, image_path: str | Path, top_k: int = 5) -> dict:
        image_result = self.predict_image(image_path, top_k=top_k)
        predicted_animal = self.text_classifier.normalize_label(str(image_result["label"]))
        text_animal = self.extract_animal(text)
        return {
            "is_match": text_animal == predicted_animal if text_animal is not None else False,
            "predicted_animal": predicted_animal,
            "text_animal": text_animal,
            "supported_animals_count": len(self.image_classifier.class_names),
            "image_result": image_result,
            "text_entities": self.text_classifier.predict_entities(text),
        }

    def verify_uploaded_image(
        self,
        text: str,
        image_bytes: bytes,
        filename: str,
        top_k: int = 5,
    ) -> dict:
        uploads_dir = self.data_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
        image_path = uploads_dir / safe_name
        image_path.write_bytes(image_bytes)
        return self.verify_details(text=text, image_path=image_path, top_k=top_k)

    def _require_prepared_path(self, key: str) -> str:
        if self.prepared_paths is None or key not in self.prepared_paths:
            raise RuntimeError("Call prepare_datasets() before training.")
        return str(self.prepared_paths[key])
