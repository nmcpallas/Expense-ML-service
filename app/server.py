from __future__ import annotations

from concurrent import futures
import logging

import grpc

from app.config import get_settings
from app.health import start_health_server
from app.model import ExpenseCategoryModel
from app.proto_loader import ensure_proto_generated
from app.service import ExpenseClassifierService
from app.storage import TrainingExample, TrainingStore
from app.tracing import TraceIdServerInterceptor, configure_logging

ensure_proto_generated()

import expense_classifier_pb2 as pb2  # noqa: E402
import expense_classifier_pb2_grpc as pb2_grpc  # noqa: E402


configure_logging()
logger = logging.getLogger(__name__)


class ExpenseClassifierGrpc(pb2_grpc.ExpenseClassifierServicer):
    def __init__(self, service: ExpenseClassifierService) -> None:
        self.service = service

    def Predict(self, request, context):
        available_categories = {
            category.id: category.name for category in request.available_categories
        }
        prediction = self.service.model.predict(
            chat_id=request.chat_id,
            raw_text=request.raw_text,
            description=request.description,
            amount=request.amount,
            available_categories=available_categories,
        )
        logger.info(
            "Prediction completed: categories=%s, needs_review=%s, model_version=%s",
            len(available_categories),
            prediction.needs_review,
            prediction.model_version,
        )
        return pb2.PredictionResponse(
            category_id=prediction.category_id,
            category_name=prediction.category_name,
            confidence=prediction.confidence,
            needs_review=prediction.needs_review,
            alternatives=[
                pb2.Alternative(
                    category_id=alternative.category_id,
                    category_name=alternative.category_name,
                    confidence=alternative.confidence,
                )
                for alternative in prediction.alternatives
            ],
            model_version=prediction.model_version,
        )

    def UploadTrainingData(self, request, context):
        examples = [
            TrainingExample(
                chat_id=example.chat_id,
                raw_text=example.raw_text,
                description=example.description,
                amount=example.amount,
                category_id=example.category_id,
                category_name=example.category_name,
                source=example.source,
            )
            for example in request.examples
        ]
        accepted, skipped = self.service.upload_training_data(
            chat_id=request.chat_id,
            replace_chat_data=request.replace_chat_data,
            examples=examples,
        )
        train_result = None
        if request.train_after_upload and accepted > 0:
            train_result = self.service.train(request.chat_id)

        logger.info(
            "Training data uploaded: accepted=%s, skipped=%s, trained=%s",
            accepted,
            skipped,
            bool(train_result and train_result.trained),
        )

        return pb2.UploadTrainingDataResponse(
            accepted_count=accepted,
            skipped_count=skipped,
            trained=bool(train_result and train_result.trained),
            message=train_result.message if train_result else "Training data uploaded.",
        )

    def Train(self, request, context):
        result = self.service.train(request.chat_id)
        logger.info(
            "Training completed: trained=%s, examples=%s, categories=%s, model_version=%s",
            result.trained,
            result.examples_count,
            result.categories_count,
            result.model_version,
        )
        return pb2.TrainResponse(
            trained=result.trained,
            examples_count=result.examples_count,
            categories_count=result.categories_count,
            model_version=result.model_version,
            message=result.message,
        )

    def Health(self, request, context):
        return pb2.HealthResponse(
            status="ok",
            trained_models_count=self.service.model.trained_models_count,
        )


def serve() -> None:
    settings = get_settings()
    store = TrainingStore(settings.data_path)
    model = ExpenseCategoryModel(
        models_dir=settings.models_dir,
        min_confidence=settings.min_confidence,
        min_examples=settings.min_examples_per_chat,
    )
    model.load_existing_models()
    service = ExpenseClassifierService(store, model)

    start_health_server(settings.health_port, service)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=[TraceIdServerInterceptor()],
    )
    pb2_grpc.add_ExpenseClassifierServicer_to_server(
        ExpenseClassifierGrpc(service), server
    )
    server.add_insecure_port(f"{settings.host}:{settings.port}")
    server.start()
    logger.info(
        "Expense ML gRPC server started: grpc=%s:%s, health=0.0.0.0:%s, trained_models=%s",
        settings.host,
        settings.port,
        settings.health_port,
        model.trained_models_count,
    )
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
