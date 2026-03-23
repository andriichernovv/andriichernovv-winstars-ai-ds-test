import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from vision.interfaces.image_classifier_interface import ImageClassifierInterface
from utils.logger import setup_logger


logger = setup_logger(__name__)


@dataclass
class ImageTrainingResult:
    model_dir: str
    class_names: list[str]
    history: dict[str, list[float]]
    num_classes: int
    best_val_accuracy: float
    test_metrics: dict[str, float]


@dataclass
class ImagePrediction:
    label: str
    confidence: float
    top_k: list[dict[str, str | float]]


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_index: dict[str, int],
        transform,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.class_to_index[str(row["label"])]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


class AnimalImageClassifier(ImageClassifierInterface):
    def __init__(
        self,
        device: str | None = None,
        image_size: int = 224,
        model_factory: Callable[[int], nn.Module] | None = None,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_size = image_size
        self.model_factory = model_factory
        self.use_pretrained = use_pretrained
        self.freeze_backbone = freeze_backbone
        self.model: nn.Module | None = None
        self.class_names: list[str] = []
        self.model_dir: Path | None = None
        self.train_transform = self._build_train_transform()
        self.eval_transform = self._build_eval_transform()

    def train(
        self,
        manifest_path: str | Path,
        model_dir: str | Path = "artifacts/image_model",
        batch_size: int = 32,
        num_epochs: int = 8,
        learning_rate: float = 3e-4,
        num_workers: int = 0,
    ) -> dict:
        manifest_path = Path(manifest_path)
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        manifest = pd.read_csv(manifest_path)
        if manifest.empty:
            raise ValueError("Image manifest is empty.")

        self.class_names = sorted(manifest["label"].unique().tolist())
        class_to_index = {label: index for index, label in enumerate(self.class_names)}
        self.model = self._create_model(len(self.class_names), pretrained=self.use_pretrained).to(self.device)
        if self.freeze_backbone:
            self._freeze_backbone(self.model)

        train_frame = manifest[manifest["split"] == "train"].copy()
        val_frame = manifest[manifest["split"] == "val"].copy()
        test_frame = manifest[manifest["split"] == "test"].copy()

        train_loader = DataLoader(
            ImageManifestDataset(train_frame, class_to_index, self.train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = None
        if not val_frame.empty:
            val_loader = DataLoader(
                ImageManifestDataset(val_frame, class_to_index, self.eval_transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(learning_rate)

        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
        best_state_dict = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        best_val_accuracy = 0.0

        for epoch_index in range(num_epochs):
            train_metrics = self._run_epoch(train_loader, criterion=criterion, optimizer=optimizer)
            val_metrics = (
                self._run_epoch(val_loader, criterion=criterion, optimizer=None)
                if val_loader is not None
                else train_metrics
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            logger.info(
                "Image epoch %s/%s train_loss=%.4f val_loss=%.4f",
                epoch_index + 1,
                num_epochs,
                train_metrics["loss"],
                val_metrics["loss"],
            )

            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_val_accuracy = val_metrics["accuracy"]
                best_state_dict = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_state_dict)
        self.model.eval()
        self.model_dir = model_dir

        test_metrics = {"loss": 0.0, "accuracy": 0.0}
        if not test_frame.empty:
            test_loader = DataLoader(
                ImageManifestDataset(test_frame, class_to_index, self.eval_transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            test_metrics = self._run_epoch(test_loader, criterion=criterion, optimizer=None)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "image_size": self.image_size,
        }
        torch.save(checkpoint, model_dir / "model.pt")
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "class_names": self.class_names,
                    "image_size": self.image_size,
                    "manifest_path": str(manifest_path),
                    "best_val_accuracy": best_val_accuracy,
                    "test_metrics": test_metrics,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return asdict(
            ImageTrainingResult(
                model_dir=str(model_dir),
                class_names=self.class_names,
                history=history,
                num_classes=len(self.class_names),
                best_val_accuracy=best_val_accuracy,
                test_metrics=test_metrics,
            )
        )

    def load(self, model_dir: str | Path = "artifacts/image_model") -> None:
        model_dir = Path(model_dir)
        checkpoint = torch.load(model_dir / "model.pt", map_location=self.device)
        self.class_names = list(checkpoint["class_names"])
        self.image_size = int(checkpoint["image_size"])
        self.train_transform = self._build_train_transform()
        self.eval_transform = self._build_eval_transform()
        self.model = self._create_model(len(self.class_names), pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.model_dir = model_dir

    def predict(self, image_path: str | Path, top_k: int = 3) -> dict:
        if self.model is None or not self.class_names:
            raise RuntimeError("Image model is not loaded.")

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        predictions: list[dict[str, str | float]] = []
        for probability, index in zip(top_probs[0], top_indices[0]):
            predictions.append(
                {
                    "label": self.class_names[int(index.item())],
                    "confidence": float(probability.item()),
                }
            )

        return asdict(
            ImagePrediction(
                label=str(predictions[0]["label"]),
                confidence=float(predictions[0]["confidence"]),
                top_k=predictions,
            )
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Image model is not initialized.")

        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                logits = self.model(images)
                loss = criterion(logits, labels)
                if is_train:
                    loss.backward()
                    optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * int(labels.size(0))
            total_correct += int((predictions == labels).sum().item())
            total_samples += int(labels.size(0))

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
        }

    def _build_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop((self.image_size, self.image_size), scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_eval_transform(self):
        return transforms.Compose([
            transforms.Resize(int(self.image_size * 1.14)),
            transforms.CenterCrop((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _create_model(self, num_classes: int, pretrained: bool) -> nn.Module:
        if self.model_factory is not None:
            return self.model_factory(num_classes)
        return self._build_default_model(num_classes, pretrained=pretrained)

    @staticmethod
    def _build_default_model(num_classes: int, pretrained: bool) -> nn.Module:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _build_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        if self.model is None:
            raise RuntimeError("Image model is not initialized.")

        classifier_parameters: list[nn.Parameter] = []
        backbone_parameters: list[nn.Parameter] = []

        if hasattr(self.model, "fc"):
            classifier_parameters = list(self.model.fc.parameters())
            layer4 = getattr(self.model, "layer4", None)
            if layer4 is not None:
                backbone_parameters = [
                    parameter for parameter in layer4.parameters()
                    if parameter.requires_grad
                ]
        elif hasattr(self.model, "classifier"):
            classifier = getattr(self.model, "classifier")
            if isinstance(classifier, nn.Module):
                classifier_parameters = list(classifier.parameters())
            features = getattr(self.model, "features", None)
            if isinstance(features, nn.Sequential) and len(features) > 0:
                backbone_parameters = [
                    parameter for parameter in features[-1].parameters()
                    if parameter.requires_grad
                ]

        if classifier_parameters:
            parameter_groups = [{"params": classifier_parameters, "lr": learning_rate}]
            if backbone_parameters:
                parameter_groups.append(
                    {
                        "params": backbone_parameters,
                        "lr": learning_rate * 0.3,
                    }
                )
            return torch.optim.AdamW(parameter_groups, weight_decay=1e-4)

        return torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=1e-4,
        )

    @staticmethod
    def _freeze_backbone(model: nn.Module) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = False

        layer4 = getattr(model, "layer4", None)
        if isinstance(layer4, nn.Module):
            for parameter in layer4.parameters():
                parameter.requires_grad = True

        if hasattr(model, "fc"):
            for parameter in model.fc.parameters():
                parameter.requires_grad = True
        elif hasattr(model, "classifier"):
            classifier = getattr(model, "classifier")
            for parameter in classifier.parameters():
                parameter.requires_grad = True
