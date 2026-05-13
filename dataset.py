import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import WeightedRandomSampler,DataLoader,Dataset
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ExoplanetDataSet(Dataset):
    def __init__(self,features,labels):
        self.X=torch.tensor(features,dtype=torch.float32).unsqueeze(1)
        self.Y=torch.tensor(labels,dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):

        return self.X[idx], self.Y[idx]
    
def get_dataloaders(file_path="exoTrain.csv",batch_size=32):
    df=pd.read_csv(file_path)
    X_raw=df.iloc[:,1:].values
    Y_raw=df.iloc[:,0].values - 1 
    
    processed_list=[]
    for row in X_raw:
        trend=savgol_filter(row,window_length=101,polyorder=2)
        processed_list.append(row-trend)
    X_processed=np.array(processed_list)


    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X_processed)

    X_train,X_val,Y_train,Y_val=train_test_split(X_scaled,Y_raw,test_size=0.2,random_state=42,stratify=Y_raw)

    class_counts=np.bincount(Y_train)
    class_weights=1/class_counts
    sample_weights=np.array([class_weights[int(label)] for label in Y_train])

    sampler=WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),num_samples=len(sample_weights),replacement=True)

    train_Dataset=ExoplanetDataSet(X_train,Y_train)
    val_dataset=ExoplanetDataSet(X_val,Y_val)

    train_loader = DataLoader(train_Dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()

    features, labels = next(iter(train_loader))
    print(f"\nBatch Features Shape: {features.shape}")
    print(f"Batch Labels Shape: {labels.shape}")