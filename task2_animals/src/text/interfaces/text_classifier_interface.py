from abc import ABC, abstractmethod
from pathlib import Path


class TextClassifierInterface(ABC):
    @abstractmethod
    def train(self, dataset_dir: str | Path, model_dir: str | Path, **kwargs) -> dict:
        pass

    @abstractmethod
    def load(self, model_dir: str | Path) -> None:
        pass

    @abstractmethod
    def predict_entities(self, text: str) -> list[dict]:
        pass

    @abstractmethod
    def extract_animal(self, text: str) -> str | None:
        pass
