# 1. Base image
FROM python:3.9

# 2. Set working directory
WORKDIR /app

# 3. Copy app and model files
COPY app/ ./app/
COPY src/model.pkl ./src/model.pkl

# 4. Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose FastAPI port
EXPOSE 8000

# 6. Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
