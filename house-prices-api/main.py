from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

from house_features import HouseFeatures

# Load your trained model
model = joblib.load("linear_reg_model.joblib")

pipeline = joblib.load("pipeline_model.pkl")
    
app = FastAPI()

@app.get("/home/")
async def get_home():
    return {"Message": "Welcome"}

@app.post("/predict/")
async def predict_price(features: HouseFeatures):
    try:
        features_dict = features.model_dump()
        
        df_features = pd.DataFrame([features_dict])
        
        X_preprocessed = pipeline.transform(df_features)
        
        # Make a prediction
        prediction = model.predict(X_preprocessed)
        
        # Return the predicted price
        return {"predicted_price": np.exp(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
