from __future__ import annotations

from app.model import ExpenseCategoryModel
from app.storage import TrainingExample, TrainingStore


class ExpenseClassifierService:
    def __init__(self, store: TrainingStore, model: ExpenseCategoryModel) -> None:
        self.store = store
        self.model = model

    def upload_training_data(
        self,
        chat_id: int,
        replace_chat_data: bool,
        examples: list[TrainingExample],
    ) -> tuple[int, int]:
        valid_examples = [
            example
            for example in examples
            if example.chat_id == chat_id
            and example.category_id
            and example.category_name
            and example.text
        ]
        skipped = len(examples) - len(valid_examples)

        if replace_chat_data:
            self.store.replace_chat_examples(chat_id, valid_examples)
        else:
            self.store.append(valid_examples)

        return len(valid_examples), skipped

    def train(self, chat_id: int):
        return self.model.train(chat_id, self.store.load(chat_id))
