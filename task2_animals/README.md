# Task 2: Animal Text-Image Verification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.40+-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![pytest](https://img.shields.io/badge/pytest-tested-blue.svg)](https://pytest.org/)

Object-oriented implementation of Task 2 with two trained models and one wrapper pipeline:

- an image model trained on the Kaggle animal dataset through `kagglehub`
- a text model trained on a synthetic labeled dataset generated inside this project
- a verification pipeline that checks whether user text matches the uploaded image

## Overview

The project solves a simple text-image consistency task:

1. train an image classifier on real animal photos
2. train a text model to extract an animal mention from user text
3. compare both predictions and return a boolean result

Example:

- text: `I think this is a tiger.`
- image: `tiger photo`
- output: `True`

## Project Structure

```text
task2_animals/
  src/
    data/
      dataset_service.py
    pipeline/
      animal_verifier.py
      verify.py
    text/
      interfaces/
        text_classifier_interface.py
      loaders/
        text_dataset_loader.py
      models/
        text_classifier.py
      train.py
      infer.py
    utils/
      logger.py
    vision/
      interfaces/
        image_classifier_interface.py
      loaders/
        animal_dataset_loader.py
      models/
        image_classifier.py
      train.py
      infer.py
  tests/
  demo.ipynb
  requirements.txt
  README.md
```

## Datasets

### Vision Dataset

- Source: `iamsouravbanerjee/animal-image-dataset-90-different-animals`
- Access method: `kagglehub`
- Storage model: images stay in the Kaggle cache, the project only builds a local CSV manifest

### Text Dataset

- Source: synthetic dataset created by `DatasetService`
- Format: JSON files for `train`, `val`, and `test`
- Labels: animal entity spans with the `ANIMAL` tag

Example training sample:

```json
{
  "id": 17,
  "text": "I think this is a tiger.",
  "animal": "tiger",
  "entities": [
    {
      "start": 20,
      "end": 26,
      "label": "ANIMAL",
      "text": "tiger"
    }
  ]
}
```

## Architecture

### `DatasetService`

Responsible for:

- loading the image dataset through `AnimalDatasetLoader`
- building the image manifest with `train`, `val`, and `test` splits
- generating the synthetic text dataset in JSON format

### `AnimalImageClassifier`

Responsible for:

- training an image classifier on the Kaggle image dataset
- using transfer learning with a pretrained `ResNet18`
- saving metrics and model artifacts
- predicting the top animal classes for a user image

### `AnimalTextClassifier`

Responsible for:

- training a token classification model on the generated text dataset
- extracting an animal entity from free-form user text
- returning detected entities and the final normalized animal label

### `AnimalVerifier`

Wrapper service responsible for:

- preparing both datasets
- training both models
- loading trained artifacts
- verifying whether user text matches the uploaded image

## Installation

From the repository root:

```bash
pip install -e .[task2,dev]
```

Or install only task-specific packages:

```bash
pip install -r task2_animals/requirements.txt
```

## Quick Start

### Notebook Workflow

Open the notebook:

```bash
jupyter notebook task2_animals/demo.ipynb
```

The notebook supports:

- training both models
- typing your own text
- uploading your own image
- running the verification pipeline from a dedicated code cell
- seeing the top image predictions and extracted text entities
- using only 10 animal classes in the demo cell to keep training time shorter
- keeping the final result formatting inside a dedicated helper function in the notebook cell
- running project logic through the project Python process to avoid common `torch` issues inside the VS Code notebook kernel

### Python API

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("task2_animals/src").resolve()))

from pipeline.animal_verifier import AnimalVerifier

verifier = AnimalVerifier()
result = verifier.fit(
    max_classes=None,
    text_samples_per_class=32,
    image_num_epochs=8,
    image_learning_rate=3e-4,
    text_num_epochs=5,
)

details = verifier.verify_details(
    text="I think this is a tiger.",
    image_path="path/to/image.jpg",
)
print(details)
```

### Result Format

`AnimalVerifier.verify_details(...)` returns a dictionary with:

- `is_match`
- `predicted_animal`
- `text_animal`
- `supported_animals_count`
- `image_result`
- `text_entities`

## Command Line Scripts

Train the image model:

```bash
python task2_animals/src/vision/train.py
```

Train the text model:

```bash
python task2_animals/src/text/train.py
```

Run image inference:

```bash
python task2_animals/src/vision/infer.py --image-path path/to/image.jpg
```

Run text inference:

```bash
python task2_animals/src/text/infer.py --text "I think this is a tiger."
```

Run full verification:

```bash
python task2_animals/src/pipeline/verify.py --text "I think this is a tiger." --image-path path/to/image.jpg --details
```

## Important Notes

- The default workflow now uses all available animal classes. It no longer trains on only the first 10 classes.
- The loader now checks the local `kagglehub` cache before attempting a network download.
- If you trained the project before this update, retrain both models. Older checkpoints may only support a small subset of animals.
- By default, `AnimalVerifier` now stores data and artifacts inside `task2_animals/data` and `task2_animals/artifacts`.
- The first image-model training run may download pretrained `torchvision` weights if they are not already cached.

## Troubleshooting

- If the notebook shows outdated keys or stale behavior after code changes, restart the kernel and rerun the import and training cells.
- If the dataset is already present in the local `kagglehub` cache, the loader will use it directly.
- If you load older checkpoints trained on a smaller class subset, the verification output can be misleading. Retrain first.
- VS Code Jupyter can duplicate widget button events. The notebook now avoids `Button.on_click` and uses a manual execution cell for verification.
- If `torch` fails during import inside the notebook kernel, use the current notebook version. It runs training and verification through the project interpreter instead of importing the full pipeline into the notebook process.

## Testing

Run the test suite:

```bash
pytest task2_animals/tests -q
```

Current local status: `7 passed`
