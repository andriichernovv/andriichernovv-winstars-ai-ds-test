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
    parser = argparse.ArgumentParser(description="Verify text against image")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--image-model-dir", type=str, default=str(TASK_ROOT / "artifacts" / "image_model"))
    parser.add_argument("--text-model-dir", type=str, default=str(TASK_ROOT / "artifacts" / "text_model"))
    parser.add_argument("--details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verifier = AnimalVerifier()
    verifier.load(
        image_model_dir=args.image_model_dir,
        text_model_dir=args.text_model_dir,
    )
    if args.details:
        print(json.dumps(verifier.verify_details(args.text, args.image_path), indent=2))
        return

    print(json.dumps(verifier.verify(args.text, args.image_path)))


if __name__ == "__main__":
    main()
