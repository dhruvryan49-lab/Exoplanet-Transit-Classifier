from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np 
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from model import TransitCNN

app=FastAPI(title="DeepSpace Exoplanet API",version="1.0")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=TransitCNN().to(device)
model.load_state_dict(torch.load("precision_weights.pth",map_location=device,weights_only=True))
model.eval()

class LightCurveInput(BaseModel):
    flux_data: list[float]

@app.post("/predict")
def predict_transit(data: LightCurveInput):
    if len(data.flux_data) != 3197:
        raise HTTPException(status_code=400, detail="Error: Input array must contain exactly 3197 time steps.")
    
    try:
        raw_flux=np.array(data.flux_data)
        trend=savgol_filter(raw_flux,window_length=101,polyorder=2)
        processed_flux=raw_flux-trend
        scaler=StandardScaler()
        scaled_flux=scaler.fit_transform(processed_flux.reshape(-1,1)).flatten()
        tensor_flux=torch.tensor(scaled_flux,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_logit=model(tensor_flux)
            probability=torch.sigmoid(raw_logit).item()*100
        
        prediction="Exoplanet Transit Detected" if probability>=50.0 else "Empty Star"

        return {
            "status": "success",
            "prediction": prediction,
            "confidence_score": round(probability, 2),
            "model_version": "TransitCNN_v1_Optimized"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")