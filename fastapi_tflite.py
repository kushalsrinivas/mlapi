from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the model from pickle file
with open("california_housing_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define input data model using Pydantic
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "California Housing Model API"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: InputData):
    # Convert the input data into a pandas DataFrame
    input_df = pd.DataFrame([data.dict().values()], columns=data.dict().keys())

    # Make prediction
    prediction = model.predict(input_df)

    # Convert the output to a Python list for JSON serialization
    prediction_list = prediction.tolist()

    return {"prediction": prediction_list}
