from app.storage import TrainingExample, TrainingStore


def example(chat_id: int, description: str, category_id: str) -> TrainingExample:
    return TrainingExample(
        chat_id=chat_id,
        raw_text=f"250 {description}",
        description=description,
        amount=250,
        category_id=category_id,
        category_name=f"category-{category_id}",
        source="TEST",
    )


def test_store_appends_and_filters_by_chat_id(tmp_path):
    store = TrainingStore(tmp_path / "training_examples.jsonl")

    store.append(
        [
            example(chat_id=1, description="кофе", category_id="food"),
            example(chat_id=2, description="такси", category_id="transport"),
        ]
    )

    assert [item.description for item in store.load(chat_id=1)] == ["кофе"]
    assert [item.description for item in store.load(chat_id=2)] == ["такси"]
    assert len(store.load()) == 2


def test_store_replaces_only_requested_chat(tmp_path):
    store = TrainingStore(tmp_path / "training_examples.jsonl")
    store.append(
        [
            example(chat_id=1, description="кофе", category_id="food"),
            example(chat_id=2, description="такси", category_id="transport"),
        ]
    )

    store.replace_chat_examples(
        chat_id=1,
        examples=[example(chat_id=1, description="обед", category_id="food")],
    )

    assert [item.description for item in store.load(chat_id=1)] == ["обед"]
    assert [item.description for item in store.load(chat_id=2)] == ["такси"]
