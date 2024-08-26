from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

import preprocessing

# Load your trained model
model = joblib.load("linear_reg_model.joblib")


# Define the columns that your model expects
class HouseFeatures(BaseModel):
    # Replace these with the actual features used by your model
    MSZoning: str
    HouseStyle: str
    LotArea: int
    YearBuilt: int
    TotRmsAbvGrd: int
    
app = FastAPI()

@app.get("/home/")
async def get_home():
    return {"Message": "Welcome"}

@app.post("/predict/")
async def predict_price(features: HouseFeatures):
    try:
        features_dict = features.model_dump()
        
        df_features = pd.DataFrame([features_dict])
        
        X_preprocessed = preprocessing.pipeline.transform(df_features)
        
        # Make a prediction
        prediction = model.predict(X_preprocessed)
        
        # Return the predicted price
        return {"predicted_price": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
