from app.model import ExpenseCategoryModel
from app.server import ExpenseClassifierGrpc, pb2
from app.service import ExpenseClassifierService
from app.storage import TrainingStore


def test_upload_train_predict_and_health_via_grpc_servicer(tmp_path):
    store = TrainingStore(tmp_path / "training_examples.jsonl")
    model = ExpenseCategoryModel(tmp_path / "models", min_confidence=0.0, min_examples=2)
    service = ExpenseClassifierService(store, model)
    servicer = ExpenseClassifierGrpc(service)

    upload_response = servicer.UploadTrainingData(
        pb2.UploadTrainingDataRequest(
            chat_id=7,
            replace_chat_data=True,
            train_after_upload=True,
            examples=[
                pb2.TrainingExample(
                    chat_id=7,
                    raw_text="250 кофе",
                    description="кофе",
                    amount=250,
                    category_id="food",
                    category_name="Кафе",
                    source="TEST",
                ),
                pb2.TrainingExample(
                    chat_id=7,
                    raw_text="700 такси",
                    description="такси",
                    amount=700,
                    category_id="transport",
                    category_name="Транспорт",
                    source="TEST",
                ),
            ],
        ),
        None,
    )

    predict_response = servicer.Predict(
        pb2.PredictRequest(
            chat_id=7,
            raw_text="300 кофе",
            description="кофе",
            amount=300,
            available_categories=[
                pb2.Category(id="food", name="Кафе"),
                pb2.Category(id="transport", name="Транспорт"),
            ],
        ),
        None,
    )
    health_response = servicer.Health(pb2.HealthRequest(), None)

    assert upload_response.accepted_count == 2
    assert upload_response.skipped_count == 0
    assert upload_response.trained is True
    assert predict_response.category_id == "food"
    assert predict_response.needs_review is False
    assert health_response.status == "ok"
    assert health_response.trained_models_count == 1
