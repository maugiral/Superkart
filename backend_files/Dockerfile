FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY superkart_model.joblib .

EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:superkart_api"]
