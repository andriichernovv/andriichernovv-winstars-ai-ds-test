from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json

from vision.models.image_classifier import AnimalImageClassifier

TASK_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image classifier inference")
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=str(TASK_ROOT / "artifacts" / "image_model"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = AnimalImageClassifier()
    model.load(args.model_dir)
    result = model.predict(args.image_path, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
