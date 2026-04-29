FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libxml2-dev libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# На Railway data/ и logs/ создаются в runtime через volume-mount
# НЕ используем VOLUME [] — Railway управляет этим сам через настройки
CMD ["python", "main.py"]
