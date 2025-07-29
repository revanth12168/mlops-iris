# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("src/model.pkl")

# Create FastAPI app
app = FastAPI()

# Define input format
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define prediction route
@app.post("/predict")
def predict(input: IrisInput):
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    prediction = model.predict(data)[0]
    return {"prediction": prediction}

