from pathlib import Path
import sys

from PIL import Image
import pytest
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def create_fake_dataset(
    root: Path,
    classes: tuple[str, ...] = ("cat", "dog"),
    images_per_class: int = 4,
    size: tuple[int, int] = (32, 32),
) -> None:
    colors = [(255, 40, 40), (40, 255, 40), (40, 40, 255), (255, 255, 40)]
    for class_index, class_name in enumerate(classes):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(images_per_class):
            image = Image.new("RGB", size, colors[class_index % len(colors)])
            image.save(class_dir / f"{class_name}_{image_index}.png")


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = outputs.flatten(start_dim=1)
        return self.classifier(outputs)


@pytest.fixture
def fake_download_dir(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "download_cache" / "animals"
    create_fake_dataset(dataset_root)
    return dataset_root.parent


@pytest.fixture
def fake_downloader(fake_download_dir: Path):
    return lambda handle: str(fake_download_dir)


@pytest.fixture
def tiny_model_factory():
    return lambda num_classes: TinyClassifier(num_classes)
