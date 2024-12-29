from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
import uvicorn, joblib
import numpy as np 
import pandas as pd

pipeline = joblib.load("pipeline.pkl")
app = FastAPI()

class Features(BaseModel):
    pclass: int = Field(ge = 1,le=3)
    sex: Literal["male", "female"]
    age: float = Field(ge=10, le=130)
    sibsp: int = Field(ge=0, le=10)
    fare: float = Field(ge=10, le=100)

    model_config = ConfigDict(extra="forbid")

pred_list={"1":[], "0":[]}


@app.post("/predictions", tags=["Model"], summary="Returns prediction of model for single data input")
async def prediction_of_model(data:Features):
    data_df = pd.DataFrame({"Pclass": data.pclass, "Sex": data.sex, "Age": data.age, "SibSp": data.sibsp, "Fare":data.fare}, index = [0,1,2,3,4,5])

    predictions = pipeline.predict(data_df)

    if int(predictions[0]) == 1:
        pred_list["1"].append(data)
    else:
        pred_list["0"].append(data)

    return {"Prediction": int(predictions[0])}

@app.get("/predictions", tags=["Model"], summary="Returns all inputed data")
async def get_all_preds():
    return {"All predictions": pred_list}

@app.get("/predictions/{pred_id}", tags=["Model"], summary = "Return data input with predicted value")
async def get_predictions(pred_id: str):
    try:
        return {"Object with prediction": pred_list[pred_id]}
    except:
        raise HTTPException(status_code=404, detail=f"Object with predicted {pred_id} was not found")

if __name__ == "__main__":
    uvicorn.run("api:app", reload=True)