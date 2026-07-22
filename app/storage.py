from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingExample:
    chat_id: int
    raw_text: str
    description: str
    amount: float
    category_id: str
    category_name: str
    source: str

    @property
    def text(self) -> str:
        return self.description.strip() or self.raw_text.strip()


class TrainingStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, chat_id: int | None = None) -> list[TrainingExample]:
        if not self.path.exists():
            return []

        examples: list[TrainingExample] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                example = TrainingExample(**row)
                if chat_id is None or example.chat_id == chat_id:
                    examples.append(example)
        return examples

    def replace_chat_examples(self, chat_id: int, examples: list[TrainingExample]) -> None:
        existing = [example for example in self.load() if example.chat_id != chat_id]
        self._write_all(existing + examples)

    def append(self, examples: list[TrainingExample]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            for example in examples:
                file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    def _write_all(self, examples: list[TrainingExample]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for example in examples:
                file.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
