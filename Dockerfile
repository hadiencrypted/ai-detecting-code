FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (including ai_model.h5 if present)
COPY . /app

# Optional: allow supplying a model URL at build time so the image can include the model.
# Usage: docker build --build-arg MODEL_URL="https://example.com/ai_model.h5" -t ai-detector:latest .
ARG MODEL_URL=""
RUN if [ -n "$MODEL_URL" ]; then \
	python - <<'PY'
import urllib.request, sys
url = """$MODEL_URL"""
print('Downloading model from', url)
urllib.request.urlretrieve(url, '/app/ai_model.h5')
print('Downloaded /app/ai_model.h5')
PY
; fi

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]
