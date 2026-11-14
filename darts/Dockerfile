FROM python:3.12-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir flask waitress numpy opencv-python-headless
COPY app.py .
COPY templates templates
CMD ["python", "app.py"]
