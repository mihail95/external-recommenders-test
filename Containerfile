FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && pip install uv

COPY pyproject.toml* setup.cfg* setup.py* requirements*.txt* /app/
COPY README.md /app/

RUN uv pip install --system -e . \
 && uv pip install --system transformers scikit-learn

COPY . /app

EXPOSE 5000

# Change the number of workers and container port here if needed
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "wsgi:app"]