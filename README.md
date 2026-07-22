# Expense ML Service

gRPC-сервис для определения категории траты по тексту из Java Telegram-bot.

## Flow

1. Java-бот получает сообщение пользователя, например `250 кофе`.
2. Java парсит сумму и описание.
3. Java вызывает `Predict` и передает `chat_id`, `raw_text`, `description`, `amount`, список категорий чата.
4. Python возвращает категорию, уверенность и альтернативы.
5. Если `confidence >= 0.8`, Java сохраняет трату автоматически.
6. Если `confidence < 0.8`, Java показывает пользователю варианты.
7. Периодически Java выгружает подтвержденные траты через `UploadTrainingData`.

## gRPC methods

- `Predict` - предсказать категорию.
- `UploadTrainingData` - загрузить обучающие примеры из Java-сервиса.
- `Train` - обучить модель для конкретного `chat_id`.
- `Health` - проверить состояние сервиса.

Контракт лежит в `proto/expense_classifier.proto`.

## Local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.server
```

По умолчанию сервис слушает `0.0.0.0:50051`.

## Tests

```bash
pytest
```

## Training data

Java-сервис может отправить данные из текущих таблиц `tg.expense` и `tg.category`:

```text
expense.chat_id
expense.description
expense.amount
expense.category_id
category.name
```

Для нового quick-flow желательно также добавить и отправлять `expense.raw_text`, например `250 кофе`.

Пример `TrainingExample`:

```json
{
  "chat_id": 123456,
  "raw_text": "250 кофе",
  "description": "кофе",
  "amount": 250,
  "category_id": "f7af2f89-5f39-4e48-893c-5d88b30aa123",
  "category_name": "Кафе",
  "source": "USER_CONFIRMED_ML"
}
```

Если `UploadTrainingDataRequest.train_after_upload=true`, сервис сразу переобучит модель для этого чата.

## Java integration notes

В Java-проект можно подключить proto через `protobuf-maven-plugin` и использовать `ExpenseClassifierGrpc.ExpenseClassifierBlockingStub`.

Перед `Predict` Java должна передавать актуальные категории чата. Это важно, потому что категории у разных чатов могут отличаться.

Если `PredictionResponse.needs_review=true`, бот должен показать `alternatives` как кнопки и дать пользователю возможность выбрать или ввести свою категорию.
