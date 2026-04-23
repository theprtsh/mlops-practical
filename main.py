from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import random
import os

app = FastAPI()

# Input data schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load models if they exist
MODELS = {}
if os.path.exists("models/model_v1.pkl") and os.path.exists("models/model_v2.pkl"):
    MODELS["v1"] = joblib.load("models/model_v1.pkl")
    MODELS["v2"] = joblib.load("models/model_v2.pkl")
else:
    print("Warning: Models not found. Run 'python data.py && python train.py' first.")

@app.get("/")
def read_root():
    return {"message": "Iris A/B Model API is running."}

@app.post("/predict")
def predict(data: IrisInput):
    if not MODELS:
        return {"error": "Models not loaded."}
    
    # Simple A/B routing (50/50)
    version = random.choice(["v1", "v2"])
    model = MODELS[version]
    
    # Pre-process input
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    
    # Predict
    prediction = int(model.predict(features)[0])
    
    return {
        "prediction": prediction,
        "version": version
    }
