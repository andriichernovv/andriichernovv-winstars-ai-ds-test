from dataclasses import dataclass
import json
from pathlib import Path
import random

from vision.loaders.animal_dataset_loader import AnimalDatasetLoader


@dataclass
class PreparedDatasetPaths:
    image_manifest_path: str
    text_dataset_dir: str
    class_names_path: str
    metadata_path: str


class DatasetService:
    def __init__(
        self,
        dataset_loader: AnimalDatasetLoader | None = None,
        data_dir: str | Path = "data",
        seed: int = 42,
    ) -> None:
        self.dataset_loader = dataset_loader or AnimalDatasetLoader(seed=seed)
        self.data_dir = Path(data_dir)
        self.seed = seed

    def prepare_datasets(
        self,
        max_classes: int | None = None,
        text_samples_per_class: int = 32,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir = self.data_dir / "manifests"
        text_dir = self.data_dir / "text_dataset"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        dataset_info = self.dataset_loader.load(
            max_classes=max_classes,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        image_manifest_path = manifests_dir / "image_manifest.csv"
        class_names_path = manifests_dir / "class_names.json"
        metadata_path = self.data_dir / "metadata.json"

        dataset_info.image_manifest.to_csv(image_manifest_path, index=False)
        class_names_path.write_text(
            json.dumps(dataset_info.class_names, indent=2),
            encoding="utf-8",
        )

        text_splits = self._build_text_dataset(
            class_names=dataset_info.class_names,
            samples_per_class=text_samples_per_class,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        for split_name, rows in text_splits.items():
            (text_dir / f"{split_name}.json").write_text(
                json.dumps(rows, indent=2),
                encoding="utf-8",
            )

        (text_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "class_names": [self.normalize_label(name) for name in dataset_info.class_names],
                    "text_samples_per_class": text_samples_per_class,
                    "splits": {key: len(value) for key, value in text_splits.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        metadata = {
            "image_root": str(dataset_info.image_root),
            "class_names": dataset_info.class_names,
            "image_manifest_path": str(image_manifest_path),
            "text_dataset_dir": str(text_dir),
            "text_samples_per_class": text_samples_per_class,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "image_manifest_path": str(image_manifest_path),
            "text_dataset_dir": str(text_dir),
            "class_names_path": str(class_names_path),
            "metadata_path": str(metadata_path),
            "class_names": dataset_info.class_names,
        }

    def _build_text_dataset(
        self,
        class_names: list[str],
        samples_per_class: int,
        val_ratio: float,
        test_ratio: float,
    ) -> dict[str, list[dict]]:
        rng = random.Random(self.seed)
        positive_templates = [
            "I think this is a {animal}.",
            "This photo shows a {animal}.",
            "It looks like a {animal}.",
            "My guess is that this is a {animal}.",
            "The animal in the picture is a {animal}.",
            "Could this be a {animal}?",
            "I am pretty sure this is a {animal}.",
            "The image probably contains a {animal}.",
        ]
        negative_templates = [
            "I cannot identify the animal in this picture.",
            "Please tell me what animal is in the image.",
            "The text does not mention any animal.",
            "I am not sure what species is shown here.",
        ]

        split_rows = {"train": [], "val": [], "test": []}
        sample_id = 0

        for raw_class_name in class_names:
            animal_name = self.normalize_label(raw_class_name)
            split_names = self._make_splits(
                size=samples_per_class + max(2, samples_per_class // 5),
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

            for split_name in split_names[:samples_per_class]:
                template = rng.choice(positive_templates)
                text = template.format(animal=animal_name)
                start = text.lower().index(animal_name.lower())
                end = start + len(animal_name)
                split_rows[split_name].append(
                    {
                        "id": sample_id,
                        "text": text,
                        "animal": animal_name,
                        "entities": [
                            {
                                "start": start,
                                "end": end,
                                "label": "ANIMAL",
                                "text": animal_name,
                            }
                        ],
                    }
                )
                sample_id += 1

            for split_name in split_names[samples_per_class:]:
                split_rows[split_name].append(
                    {
                        "id": sample_id,
                        "text": rng.choice(negative_templates),
                        "animal": "",
                        "entities": [],
                    }
                )
                sample_id += 1

        return split_rows

    @staticmethod
    def _make_splits(size: int, val_ratio: float, test_ratio: float) -> list[str]:
        if size <= 0:
            return []

        n_test = int(size * test_ratio)
        n_val = int(size * val_ratio)
        if size >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)

        while size - n_test - n_val < 1 and n_val > 0:
            n_val -= 1
        while size - n_test - n_val < 1 and n_test > 0:
            n_test -= 1

        n_train = size - n_val - n_test
        return (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)

    @staticmethod
    def normalize_label(value: str) -> str:
        return " ".join(value.replace("_", " ").replace("-", " ").split()).lower()
