from __future__ import annotations

import argparse

from app.config import get_settings
from app.model import ExpenseCategoryModel
from app.storage import TrainingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train expense classifier.")
    parser.add_argument("--chat-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    store = TrainingStore(settings.data_path)
    model = ExpenseCategoryModel(
        models_dir=settings.models_dir,
        min_confidence=settings.min_confidence,
        min_examples=settings.min_examples_per_chat,
    )
    result = model.train(args.chat_id, store.load(args.chat_id))
    print(result.message)
    print(
        f"trained={result.trained}, examples={result.examples_count}, "
        f"categories={result.categories_count}, version={result.model_version}"
    )


if __name__ == "__main__":
    main()
