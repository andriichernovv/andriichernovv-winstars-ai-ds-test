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
    parser = argparse.ArgumentParser(description="Train text classifier for Task 2")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--text-samples-per-class", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default=str(TASK_ROOT / "data"))
    parser.add_argument("--model-dir", type=str, default=str(TASK_ROOT / "artifacts" / "text_model"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verifier = AnimalVerifier(data_dir=args.data_dir)
    prepared = verifier.prepare_datasets(
        max_classes=args.max_classes,
        text_samples_per_class=args.text_samples_per_class,
    )
    result = verifier.train_text(
        text_dataset_dir=prepared["text_dataset_dir"],
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
