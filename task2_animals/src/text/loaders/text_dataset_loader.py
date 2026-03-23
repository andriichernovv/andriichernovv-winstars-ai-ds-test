import json
from pathlib import Path


class TextDatasetLoader:
    def load_split(self, dataset_dir: str | Path, split_name: str) -> list[dict]:
        dataset_dir = Path(dataset_dir)
        path = dataset_dir / f"{split_name}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def load_metadata(self, dataset_dir: str | Path) -> dict:
        dataset_dir = Path(dataset_dir)
        return json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
