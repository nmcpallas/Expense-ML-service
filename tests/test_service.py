from app.model import ExpenseCategoryModel
from app.service import ExpenseClassifierService
from app.storage import TrainingExample, TrainingStore


def make_example(
    chat_id: int,
    description: str,
    category_id: str = "food",
    category_name: str = "Кафе",
) -> TrainingExample:
    return TrainingExample(
        chat_id=chat_id,
        raw_text=f"250 {description}",
        description=description,
        amount=250,
        category_id=category_id,
        category_name=category_name,
        source="TEST",
    )


def test_upload_training_data_accepts_only_valid_examples_for_chat(tmp_path):
    store = TrainingStore(tmp_path / "training_examples.jsonl")
    model = ExpenseCategoryModel(tmp_path / "models", min_confidence=0.8, min_examples=2)
    service = ExpenseClassifierService(store, model)

    accepted, skipped = service.upload_training_data(
        chat_id=1,
        replace_chat_data=False,
        examples=[
            make_example(chat_id=1, description="кофе"),
            make_example(chat_id=2, description="такси"),
            make_example(chat_id=1, description="", category_id=""),
        ],
    )

    assert accepted == 1
    assert skipped == 2
    assert [item.description for item in store.load(chat_id=1)] == ["кофе"]


def test_train_delegates_to_model_with_chat_examples(tmp_path):
    store = TrainingStore(tmp_path / "training_examples.jsonl")
    model = ExpenseCategoryModel(tmp_path / "models", min_confidence=0.0, min_examples=2)
    service = ExpenseClassifierService(store, model)
    service.upload_training_data(
        chat_id=1,
        replace_chat_data=False,
        examples=[
            make_example(chat_id=1, description="кофе", category_id="food", category_name="Кафе"),
            make_example(chat_id=1, description="такси", category_id="transport", category_name="Транспорт"),
        ],
    )

    result = service.train(chat_id=1)

    assert result.trained is True
    assert result.examples_count == 2
    assert result.categories_count == 2
    assert model.trained_models_count == 1
