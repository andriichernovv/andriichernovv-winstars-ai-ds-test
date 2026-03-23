from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json

from text.models.text_classifier import AnimalTextClassifier

TASK_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text classifier inference")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=str(TASK_ROOT / "artifacts" / "text_model"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AnimalTextClassifier()
    model.load(args.model_dir)
    result = {
        "entities": model.predict_entities(args.text),
        "animal": model.extract_animal(args.text),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
