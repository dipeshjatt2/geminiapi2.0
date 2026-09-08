FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    TMP_DIR=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data

EXPOSE 8787

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8787}"]
