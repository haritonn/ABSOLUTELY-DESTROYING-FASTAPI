from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn, joblib
import numpy as np 
import pandas as pd

pipeline = joblib.load("pipeline.pkl")
app = FastAPI()

class Features(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    fare: float 

@app.post("/predictions", tags=["Model"], summary="Returns prediction of model for single data input")
async def prediction_of_model(data:Features):
    data_df = pd.DataFrame({"Pclass": data.pclass, "Sex": data.sex, "Age": data.age, "SibSp": data.sibsp, "Fare":data.fare}, index = [0,1,2,3,4,5])

    predictions = pipeline.predict(data_df)
    return {"Prediction": int(predictions[0])}


if __name__ == "__main__":
    uvicorn.run("api:app", reload=True)