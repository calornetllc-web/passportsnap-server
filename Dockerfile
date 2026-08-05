FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download the model at build time so the first request isn't slow
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

# Cloud Run sets $PORT at runtime; shell-form CMD lets us expand it
CMD exec gunicorn --bind :$PORT --workers=1 --threads=4 --timeout=0 app:app
