from app.model import ExpenseCategoryModel
from app.storage import TrainingExample


def make_example(
    description: str,
    category_id: str,
    category_name: str,
    amount: float = 250,
) -> TrainingExample:
    return TrainingExample(
        chat_id=42,
        raw_text=f"{amount} {description}",
        description=description,
        amount=amount,
        category_id=category_id,
        category_name=category_name,
        source="TEST",
    )


def test_train_rejects_too_few_examples(tmp_path):
    model = ExpenseCategoryModel(tmp_path / "models", min_confidence=0.8, min_examples=3)

    result = model.train(
        chat_id=42,
        examples=[
            make_example("кофе", "food", "Кафе"),
            make_example("такси", "transport", "Транспорт"),
        ],
    )

    assert result.trained is False
    assert result.examples_count == 2
    assert "Need at least 3 examples" in result.message


def test_train_predict_and_load_existing_model(tmp_path):
    models_dir = tmp_path / "models"
    model = ExpenseCategoryModel(models_dir, min_confidence=0.0, min_examples=2)
    examples = [
        make_example("кофе", "food", "Кафе"),
        make_example("капучино", "food", "Кафе"),
        make_example("латте", "food", "Кафе"),
        make_example("такси", "transport", "Транспорт", amount=900),
        make_example("метро", "transport", "Транспорт", amount=70),
        make_example("автобус", "transport", "Транспорт", amount=50),
    ]

    train_result = model.train(chat_id=42, examples=examples)
    prediction = model.predict(
        chat_id=42,
        raw_text="300 кофе",
        description="кофе",
        amount=300,
        available_categories={"food": "Кафе", "transport": "Транспорт"},
    )

    assert train_result.trained is True
    assert prediction.needs_review is False
    assert prediction.category_id == "food"
    assert prediction.category_name == "Кафе"
    assert prediction.confidence > 0
    assert prediction.alternatives

    reloaded = ExpenseCategoryModel(models_dir, min_confidence=0.0, min_examples=2)
    reloaded.load_existing_models()

    assert reloaded.trained_models_count == 1
    assert reloaded.predict(
        chat_id=42,
        raw_text="80 метро",
        description="метро",
        amount=80,
        available_categories={"food": "Кафе", "transport": "Транспорт"},
    ).category_id == "transport"


def test_predict_returns_review_when_model_is_missing(tmp_path):
    model = ExpenseCategoryModel(tmp_path / "models", min_confidence=0.8, min_examples=2)

    prediction = model.predict(
        chat_id=100,
        raw_text="250 кофе",
        description="кофе",
        amount=250,
        available_categories={"food": "Кафе"},
    )

    assert prediction.needs_review is True
    assert prediction.category_id == ""
    assert prediction.confidence == 0.0
