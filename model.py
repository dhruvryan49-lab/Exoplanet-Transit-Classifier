import torch
import torch.nn as nn   


class TransitCNN(nn.Module):
    def __init__(self):
        super(TransitCNN,self).__init__()
        self.conv1=nn.Conv1d(in_channels=1,out_channels=16,kernel_size=5,padding=2)
        self.bn1=nn.BatchNorm1d(16)
        self.relu1=nn.ReLU()
        self.pool1= nn.MaxPool1d(kernel_size=4)

        self.conv2=nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(64*49,128)
        self.relu4=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self.fc2=nn.Linear(128,1)

    def forward(self,x):
        
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x=self.flatten(x)
        x=self.relu4(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x

if __name__=="__main__":
    dummy=torch.randn(32,1,3197)

    model=TransitCNN()
    output=model(dummy)
    print(dummy.shape)
    print(output.shape)
    if output.shape==torch.Size([32,1]):
        print("cum")

