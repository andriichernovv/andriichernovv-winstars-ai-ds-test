from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable

import kagglehub
import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class AnimalDatasetInfo:
    image_root: Path
    class_names: list[str]
    image_manifest: pd.DataFrame


class AnimalDatasetLoader:
    DATASET_HANDLE = "iamsouravbanerjee/animal-image-dataset-90-different-animals"

    def __init__(
        self,
        dataset_handle: str | None = None,
        dataset_downloader: Callable[[str], str] | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset_handle = dataset_handle or self.DATASET_HANDLE
        self.dataset_downloader = dataset_downloader or kagglehub.dataset_download
        self.use_local_cache = dataset_downloader is None
        self.seed = seed

    def load(
        self,
        max_classes: int | None = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> AnimalDatasetInfo:
        dataset_root = self._resolve_dataset_root()
        image_root = self._find_image_root(dataset_root)
        class_dirs = self._list_class_dirs(image_root)
        if max_classes is not None:
            class_dirs = class_dirs[:max_classes]

        class_names = [path.name for path in class_dirs]
        manifest = self._build_manifest(class_dirs, val_ratio=val_ratio, test_ratio=test_ratio)
        return AnimalDatasetInfo(
            image_root=image_root,
            class_names=class_names,
            image_manifest=manifest,
        )

    def _build_manifest(
        self,
        class_dirs: list[Path],
        val_ratio: float,
        test_ratio: float,
    ) -> pd.DataFrame:
        rng = random.Random(self.seed)
        rows: list[dict] = []

        for class_dir in class_dirs:
            image_paths = sorted(
                [
                    path for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda path: path.name.lower(),
            )
            rng.shuffle(image_paths)
            splits = self._make_splits(len(image_paths), val_ratio=val_ratio, test_ratio=test_ratio)

            for image_path, split_name in zip(image_paths, splits):
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": class_dir.name,
                        "split": split_name,
                    }
                )

        return pd.DataFrame(rows)

    def _resolve_dataset_root(self) -> Path:
        if self.use_local_cache:
            cached_root = self._resolve_cached_dataset_root()
            if cached_root is not None:
                return cached_root
        return Path(self.dataset_downloader(self.dataset_handle))

    def _resolve_cached_dataset_root(self) -> Path | None:
        handle_parts = self.dataset_handle.strip("/").split("/")
        if len(handle_parts) != 2:
            return None

        owner, dataset_name = handle_parts
        versions_root = (
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / owner
            / dataset_name
            / "versions"
        )
        if not versions_root.exists():
            return None

        version_dirs = [path for path in versions_root.iterdir() if path.is_dir()]
        if not version_dirs:
            return None

        def sort_key(path: Path) -> tuple[int, str]:
            try:
                return (int(path.name), path.name)
            except ValueError:
                return (-1, path.name)

        return sorted(version_dirs, key=sort_key)[-1]

    @staticmethod
    def _find_image_root(dataset_path: Path) -> Path:
        candidates = [dataset_path]
        for path in dataset_path.rglob("*"):
            if path.is_dir() and len(path.relative_to(dataset_path).parts) <= 3:
                candidates.append(path)

        best_candidate = None
        best_score = -1

        for candidate in candidates:
            class_dirs = [
                path for path in candidate.iterdir()
                if path.is_dir()
                and any(
                    file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
                    for file_path in path.iterdir()
                )
            ]
            if len(class_dirs) > best_score:
                best_candidate = candidate
                best_score = len(class_dirs)

        if best_candidate is None or best_score <= 0:
            raise FileNotFoundError(f"Could not find image classes under {dataset_path}")

        return best_candidate

    @staticmethod
    def _list_class_dirs(image_root: Path) -> list[Path]:
        class_dirs = [
            path for path in image_root.iterdir()
            if path.is_dir()
            and any(
                file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
                for file_path in path.iterdir()
            )
        ]
        class_dirs.sort(key=lambda path: path.name.lower())
        if not class_dirs:
            raise ValueError("No image classes found.")
        return class_dirs

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
