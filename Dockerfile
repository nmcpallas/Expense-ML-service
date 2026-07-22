FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY proto proto
COPY train.py .

EXPOSE 50051
EXPOSE 8080

CMD ["python", "-m", "app.server"]
