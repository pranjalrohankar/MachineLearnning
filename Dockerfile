FROM python:3.12-slim

# Install system dependencies needed for scapy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "wsgi:app"]
