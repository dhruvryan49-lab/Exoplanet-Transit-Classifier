import torch
import torch.nn as nn 
import torch.optim as optim
import random
import numpy as np
from dataset import get_dataloaders
from model import TransitCNN
from sklearn.metrics import classification_report, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train_model():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=TransitCNN().to(device)
    train_loader,val_loader=get_dataloaders()
    class_penalty=torch.tensor([7.0]).to(device)

    criterion=nn.BCEWithLogitsLoss(pos_weight=class_penalty)
    optimizer=optim.Adam(model.parameters(),lr=0.0005,weight_decay=1e-4)


    epochs=30
    for epoch in range(epochs):
        model.train()
        running_loss=0.0

        for batch_idx, (features,labels) in enumerate(train_loader):
            features,labels=features.to(device),labels.to(device)
            optimizer.zero_grad()
            predictions=model(features)
            loss=criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()

        avg_loss=running_loss/len(train_loader)
        print(f"average training loss,{avg_loss:.4f}")
    print("cum")

    model.eval()
    all_preds=[]
    all_true_labels=[]

    with torch.no_grad():
        for features, labels in val_loader:
            features,labels=features.to(device),labels.to(device)
            raw_outputs=model(features)
            probabilities=torch.sigmoid(raw_outputs)
            predicted_classes=(probabilities>=0.5).float()
            all_preds.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    print(confusion_matrix(all_true_labels, all_preds))
    print(classification_report(all_true_labels, all_preds))    
    torch.save(model.state_dict(), "precision_weights.pth")       

if __name__=="__main__":
    train_model()

# torch.save(model.state_dict(), "precision_weights.pth")



