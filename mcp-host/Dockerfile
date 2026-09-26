FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Critical: let Python import from src/
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "api.apiserver:app", "--host", "0.0.0.0", "--port", "8000"]