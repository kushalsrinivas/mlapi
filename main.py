from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the model from the pickle file
with open("crop_detection_model.pkl", "rb") as file:
    model = pickle.load(file)


# Define input data model using Pydantic


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Pickle Model API"}


# Prediction endpoint
# Define input data model using Pydantic
class InputData(BaseModel):
    temp_min: float
    temp_max: float
    rainfall_min: float
    rainfall_max: float
    humidity_min: float
    humidity_max: float
    sunlight_min: float
    sunlight_max: float


# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert the input data into a pandas DataFrame
    input_df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    # Make prediction using the loaded model
    prediction = model.predict(input_df)

    # Convert the output to a Python list for JSON serialization
    prediction_list = prediction.tolist()

    return {"input": data.dict(), "prediction": prediction_list}
