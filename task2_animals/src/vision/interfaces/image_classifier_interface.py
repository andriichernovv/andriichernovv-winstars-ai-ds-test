from abc import ABC, abstractmethod
from pathlib import Path


class ImageClassifierInterface(ABC):
    @abstractmethod
    def train(self, manifest_path: str | Path, model_dir: str | Path, **kwargs) -> dict:
        pass

    @abstractmethod
    def load(self, model_dir: str | Path) -> None:
        pass

    @abstractmethod
    def predict(self, image_path: str | Path, top_k: int = 3) -> dict:
        pass
