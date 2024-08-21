from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

import preprocessing

# Load your trained model
model = joblib.load(".\linear_reg_model.joblib")

# Define the columns that your model expects
class HouseFeatures(BaseModel):
    # Replace these with the actual features used by your model
    MSSubClass: int
    LotFrontage: float
    LotArea: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    MasVnrArea: float
    TotalBsmtSF: int
    GrLivArea: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    TotRmsAbvGrd: int
    Fireplaces: int
    GarageCars: int
    GarageArea: int
    WoodDeckSF: int
    OpenPorchSF: int
    # Add all other necessary features

app = FastAPI()

@app.post("/predict/")
async def predict_price(features: HouseFeatures):
    
    try:
        # Convert the features to a numpy array in the same order as your model expects
        feature_array = np.array([[features.MSSubClass, features.LotFrontage, features.LotArea,
                                   features.OverallQual, features.OverallCond, features.YearBuilt,
                                   features.YearRemodAdd, features.MasVnrArea, features.TotalBsmtSF,
                                   features.GrLivArea, features.FullBath, features.HalfBath,
                                   features.BedroomAbvGr, features.KitchenAbvGr, features.TotRmsAbvGrd,
                                   features.Fireplaces, features.GarageCars, features.GarageArea,
                                   features.WoodDeckSF, features.OpenPorchSF]])
        
    
        X_preprocessed = preprocessing.pipeline.fit_transform(feature_array)
    
        # Make a prediction
        prediction = model.predict(X_preprocessed)
        
        # Return the predicted price
        return {"predicted_price": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
