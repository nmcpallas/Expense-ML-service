from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.storage import TrainingExample


@dataclass(frozen=True)
class CategoryCandidate:
    category_id: str
    category_name: str
    confidence: float


@dataclass(frozen=True)
class Prediction:
    category_id: str
    category_name: str
    confidence: float
    needs_review: bool
    alternatives: list[CategoryCandidate]
    model_version: str


@dataclass(frozen=True)
class TrainResult:
    trained: bool
    examples_count: int
    categories_count: int
    model_version: str
    message: str


class ExpenseCategoryModel:
    def __init__(self, models_dir: Path, min_confidence: float, min_examples: int) -> None:
        self.models_dir = models_dir
        self.min_confidence = min_confidence
        self.min_examples = min_examples
        self._models: dict[int, dict] = {}

    def load_existing_models(self) -> None:
        if not self.models_dir.exists():
            return
        for model_path in self.models_dir.glob("chat_*.joblib"):
            chat_id = int(model_path.stem.removeprefix("chat_"))
            self._models[chat_id] = joblib.load(model_path)

    @property
    def trained_models_count(self) -> int:
        return len(self._models)

    def train(self, chat_id: int, examples: list[TrainingExample]) -> TrainResult:
        clean_examples = [
            example
            for example in examples
            if example.text and example.category_id and example.category_name
        ]
        categories = {example.category_id for example in clean_examples}
        if len(clean_examples) < self.min_examples:
            return TrainResult(
                trained=False,
                examples_count=len(clean_examples),
                categories_count=len(categories),
                model_version="",
                message=f"Need at least {self.min_examples} examples for chat {chat_id}.",
            )
        if len(categories) < 2:
            return TrainResult(
                trained=False,
                examples_count=len(clean_examples),
                categories_count=len(categories),
                model_version="",
                message="Need at least two categories.",
            )

        labels = [example.category_id for example in clean_examples]
        category_names = {
            example.category_id: example.category_name for example in clean_examples
        }
        texts = [self._build_feature_text(example) for example in clean_examples]

        pipeline = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        lowercase=True,
                        min_df=1,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                    ),
                ),
            ]
        )
        pipeline.fit(texts, labels)

        model_version = str(int(time()))
        payload = {
            "pipeline": pipeline,
            "category_names": category_names,
            "model_version": model_version,
        }
        self.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self._model_path(chat_id))
        self._models[chat_id] = payload

        return TrainResult(
            trained=True,
            examples_count=len(clean_examples),
            categories_count=len(categories),
            model_version=model_version,
            message="Model trained.",
        )

    def predict(
        self,
        chat_id: int,
        raw_text: str,
        description: str,
        amount: float,
        available_categories: dict[str, str],
    ) -> Prediction:
        model = self._models.get(chat_id)
        if model is None:
            return Prediction("", "", 0.0, True, [], "")

        feature_text = self._build_feature_text(
            TrainingExample(
                chat_id=chat_id,
                raw_text=raw_text,
                description=description,
                amount=amount,
                category_id="",
                category_name="",
                source="predict",
            )
        )
        pipeline = model["pipeline"]
        probabilities = pipeline.predict_proba([feature_text])[0]
        classes = list(pipeline.named_steps["classifier"].classes_)

        ranked = sorted(
            zip(classes, probabilities, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        candidates: list[CategoryCandidate] = []
        for category_id, confidence in ranked:
            category_name = available_categories.get(category_id) or model[
                "category_names"
            ].get(category_id, "")
            if available_categories and category_id not in available_categories:
                continue
            candidates.append(
                CategoryCandidate(
                    category_id=category_id,
                    category_name=category_name,
                    confidence=round(float(confidence), 4),
                )
            )

        if not candidates:
            return Prediction("", "", 0.0, True, [], model["model_version"])

        best = candidates[0]
        needs_review = best.confidence < self.min_confidence
        return Prediction(
            category_id=best.category_id if not needs_review else "",
            category_name=best.category_name if not needs_review else "",
            confidence=best.confidence,
            needs_review=needs_review,
            alternatives=candidates[:3],
            model_version=model["model_version"],
        )

    def _model_path(self, chat_id: int) -> Path:
        return self.models_dir / f"chat_{chat_id}.joblib"

    @staticmethod
    def _build_feature_text(example: TrainingExample) -> str:
        amount_bucket = ExpenseCategoryModel._amount_bucket(example.amount)
        return f"{example.raw_text} {example.description} amount_{amount_bucket}".strip()

    @staticmethod
    def _amount_bucket(amount: float) -> str:
        if amount <= 0 or np.isnan(amount):
            return "unknown"
        if amount < 100:
            return "small"
        if amount < 1_000:
            return "medium"
        if amount < 10_000:
            return "large"
        return "very_large"
