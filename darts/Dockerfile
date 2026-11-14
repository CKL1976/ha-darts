FROM python:3.12-slim-bookworm
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir flask waitress numpy scipy opencv-python-headless "paho-mqtt>=2,<3"
COPY app.py camera.py mqtt_entities.py diagnostics.py ./
COPY templates templates
CMD ["python", "app.py"]
