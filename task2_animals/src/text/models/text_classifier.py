import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, BertConfig, BertForTokenClassification, BertTokenizerFast

from text.interfaces.text_classifier_interface import TextClassifierInterface
from text.loaders.text_dataset_loader import TextDatasetLoader
from utils.logger import setup_logger


logger = setup_logger(__name__)

LABEL_TO_ID = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


@dataclass
class TextTrainingResult:
    model_dir: str
    class_names: list[str]
    history: dict[str, list[float]]


class TextEntityDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer: BertTokenizerFast, max_length: int) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        encoding = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        special_tokens_mask = encoding["special_tokens_mask"].squeeze(0).tolist()
        labels: list[int] = []

        for offset, is_special in zip(offsets, special_tokens_mask):
            if is_special:
                labels.append(-100)
                continue

            token_start, token_end = int(offset[0]), int(offset[1])
            label_id = LABEL_TO_ID["O"]

            for entity in sample["entities"]:
                entity_start = int(entity["start"])
                entity_end = int(entity["end"])
                if token_start >= entity_start and token_end <= entity_end:
                    label_id = LABEL_TO_ID["B-ANIMAL"] if token_start == entity_start else LABEL_TO_ID["I-ANIMAL"]
                    break

            labels.append(label_id)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class AnimalTextClassifier(TextClassifierInterface):
    def __init__(
        self,
        dataset_loader: TextDatasetLoader | None = None,
        device: str | None = None,
        max_length: int = 64,
    ) -> None:
        self.dataset_loader = dataset_loader or TextDatasetLoader()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_length = max_length
        self.model: nn.Module | None = None
        self.tokenizer: BertTokenizerFast | None = None
        self.class_names: list[str] = []
        self.model_dir: Path | None = None

    def train(
        self,
        dataset_dir: str | Path,
        model_dir: str | Path = "artifacts/text_model",
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 5e-4,
        hidden_size: int = 64,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 128,
    ) -> dict:
        dataset_dir = Path(dataset_dir)
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        train_samples = self.dataset_loader.load_split(dataset_dir, "train")
        val_samples = self.dataset_loader.load_split(dataset_dir, "val")
        test_samples = self.dataset_loader.load_split(dataset_dir, "test")
        metadata = self.dataset_loader.load_metadata(dataset_dir)
        self.class_names = list(metadata.get("class_names", []))

        all_samples = train_samples + val_samples + test_samples
        vocab = self._build_vocabulary(all_samples)
        vocab_path = model_dir / "vocab.txt"
        vocab_path.write_text("\n".join(vocab), encoding="utf-8")

        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max(self.max_length + 2, 128),
            num_labels=len(LABEL_TO_ID),
            label2id=LABEL_TO_ID,
            id2label=ID_TO_LABEL,
        )
        self.model = BertForTokenClassification(config).to(self.device)

        train_loader = DataLoader(
            TextEntityDataset(train_samples, self.tokenizer, self.max_length),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = None
        if val_samples:
            val_loader = DataLoader(
                TextEntityDataset(val_samples, self.tokenizer, self.max_length),
                batch_size=batch_size,
                shuffle=False,
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        history = {"train_loss": [], "val_loss": []}
        best_state_dict = self.model.state_dict()
        best_val_loss = float("inf")

        for epoch_index in range(num_epochs):
            train_loss = self._run_epoch(train_loader, optimizer=optimizer)
            val_loss = self._run_epoch(val_loader, optimizer=None) if val_loader is not None else train_loss
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            logger.info(
                "Text epoch %s/%s train_loss=%.4f val_loss=%.4f",
                epoch_index + 1,
                num_epochs,
                train_loss,
                val_loss,
            )

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        self.model.load_state_dict(best_state_dict)
        self.model.eval()
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        (model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "class_names": self.class_names,
                    "max_length": self.max_length,
                    "dataset_dir": str(dataset_dir),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.model_dir = model_dir

        return asdict(
            TextTrainingResult(
                model_dir=str(model_dir),
                class_names=self.class_names,
                history=history,
            )
        )

    def load(self, model_dir: str | Path = "artifacts/text_model") -> None:
        model_dir = Path(model_dir)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.class_names = list(metadata.get("class_names", []))
        self.max_length = int(metadata.get("max_length", self.max_length))
        self.model_dir = model_dir

    def predict_entities(self, text: str) -> list[dict]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text model is not loaded.")

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        special_tokens_mask = encoding["special_tokens_mask"].squeeze(0).tolist()
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0)
            predicted_labels = torch.argmax(probabilities, dim=-1).tolist()

        entities: list[dict[str, Any]] = []
        current_tokens: list[tuple[int, int, float]] = []

        for label_id, offset, is_special, token_scores in zip(
            predicted_labels,
            offsets,
            special_tokens_mask,
            probabilities.tolist(),
        ):
            if is_special:
                if current_tokens:
                    entities.append(self._finalize_entity(text, current_tokens))
                    current_tokens = []
                continue

            token_start, token_end = int(offset[0]), int(offset[1])
            if token_start == token_end:
                continue

            label_name = ID_TO_LABEL.get(int(label_id), "O")
            if label_name == "B-ANIMAL":
                if current_tokens:
                    entities.append(self._finalize_entity(text, current_tokens))
                current_tokens = [(token_start, token_end, float(token_scores[label_id]))]
            elif label_name == "I-ANIMAL" and current_tokens:
                current_tokens.append((token_start, token_end, float(token_scores[label_id])))
            else:
                if current_tokens:
                    entities.append(self._finalize_entity(text, current_tokens))
                    current_tokens = []

        if current_tokens:
            entities.append(self._finalize_entity(text, current_tokens))

        return entities

    def extract_animal(self, text: str) -> str | None:
        normalized_entities: list[str] = []
        for entity in self.predict_entities(text):
            animal_name = self.normalize_label(str(entity["text"]))
            if animal_name in self.class_names and animal_name not in normalized_entities:
                normalized_entities.append(animal_name)

        if normalized_entities:
            return normalized_entities[0]

        normalized_text = self.normalize_label(text)
        for animal_name in self.class_names:
            if animal_name and animal_name in normalized_text:
                return animal_name

        return None

    def _run_epoch(
        self,
        loader: DataLoader | None,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        if self.model is None:
            raise RuntimeError("Text model is not initialized.")
        if loader is None:
            return 0.0

        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if is_train:
                optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return total_loss / max(total_batches, 1)

    @staticmethod
    def _build_vocabulary(samples: list[dict]) -> list[str]:
        pattern = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]")
        tokens = set(SPECIAL_TOKENS)
        for sample in samples:
            for token in pattern.findall(sample["text"].lower()):
                tokens.add(token)
        return SPECIAL_TOKENS + sorted(token for token in tokens if token not in SPECIAL_TOKENS)

    @staticmethod
    def _finalize_entity(text: str, token_spans: list[tuple[int, int, float]]) -> dict[str, Any]:
        start = token_spans[0][0]
        end = token_spans[-1][1]
        score = sum(token_score for _, _, token_score in token_spans) / len(token_spans)
        return {
            "text": text[start:end],
            "start": start,
            "end": end,
            "label": "ANIMAL",
            "score": score,
        }

    @staticmethod
    def normalize_label(value: str) -> str:
        return " ".join(value.replace("_", " ").replace("-", " ").split()).lower()
