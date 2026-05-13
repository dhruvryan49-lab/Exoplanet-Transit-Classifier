import torch
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from model import TransitCNN

def plot_prediction():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df=pd.read_csv("exoTest.csv")
    exoplanet_rows=test_df[test_df.iloc[:,0]==2]

    if len(exoplanet_rows)==0:
        print("No exoplanet")
        return 
    raw_flux= exoplanet_rows.iloc[0,1:].values
    trend=savgol_filter(raw_flux,window_length=101,polyorder=2)
    processed_flux=raw_flux-trend
    
    scaler=StandardScaler()
    scaled_flux=scaler.fit_transform(processed_flux.reshape(1,-1)).flatten()
    
    tensor_flux=torch.tensor(scaled_flux,dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    model=TransitCNN().to(device)

    model.load_state_dict(torch.load("precision_weights.pth",map_location=device))
    model.eval()

    with torch.no_grad():
        raw_logit=model(tensor_flux)
        probability=torch.sigmoid(raw_logit).item()*100
    print(f"CNN Exoplanet Confidence: {probability:.2f}%")
    print("Rendering plot...")
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 5))

    plt.plot(raw_flux, label="Raw Light Curve", color='#4a4a4a', alpha=0.7)
    plt.plot(trend, label="Extracted Trend (Savitzky-Golay)", color='#00ffcc', linewidth=2)

    plt.title(f"DeepSpace-CNN Inference\nExoplanet Detection Confidence: {probability:.2f}%", fontsize=14, pad=15)
    plt.xlabel("Observation Time Steps")
    plt.ylabel("Stellar Flux")
    plt.legend(loc="upper right")
    plt.grid(color='#333333', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    
    plt.savefig("detection_graph.png", dpi=300)
    print("Graph saved successfully as detection_graph.png")
    plt.show()

if __name__ == "__main__":
    plot_prediction()
