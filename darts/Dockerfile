FROM python:3.12-slim-bookworm
WORKDIR /app
RUN pip install --no-cache-dir flask waitress numpy scipy pillow scikit-image==0.25.2 "paho-mqtt>=2,<3"
COPY app.py camera.py mqtt_entities.py diagnostics.py imaging.py ./
COPY templates templates
CMD ["python", "app.py"]
