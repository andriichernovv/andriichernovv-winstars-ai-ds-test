from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json

from pipeline.animal_verifier import AnimalVerifier

TASK_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier for Task 2")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--text-samples-per-class", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default=str(TASK_ROOT / "data"))
    parser.add_argument("--model-dir", type=str, default=str(TASK_ROOT / "artifacts" / "image_model"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verifier = AnimalVerifier(data_dir=args.data_dir)
    prepared = verifier.prepare_datasets(
        max_classes=args.max_classes,
        text_samples_per_class=args.text_samples_per_class,
    )
    result = verifier.train_image(
        image_manifest_path=prepared["image_manifest_path"],
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
