FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# RUN apt-get update && apt-get install -y --no-install-recommends openssl \
#     && mkdir -p /certs \
#     && openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
#        -keyout /certs/key.pem \
#        -out /certs/cert.pem \
#        -subj "/CN=localhost" \
#     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch
# RUN pip install --no-cache-dir torch torchvision \
#     --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "--certfile", "/certs/cert.pem", "--keyfile", "/certs/key.pem", "app:app"]