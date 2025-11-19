import torch.nn as nn
import torch

class ResBk(nn.Module):
    def __init__(self,in_channels,out_channels,stride =1):
        super().__init__()
        self.conv1 =nn.Conv2d(in_channels,out_channels,3,stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride,padding=1,bias=False)
        self.bn2 =  nn.BatchNorm2d(out_channels)
        ##Shortcut Logic here
        self.shortcut = nn.Sequential()
        self.use_skt = stride !=1 or in_channels != out_channels
        if self.use_skt:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,stride=stride,bias=False),nn.BatchNorm2d(out_channels))
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        shortcut =self.shortcut(x) if self.use_skt else x
        out_add = out+shortcut
        out = torch.relu(out_add)
        return out_add
    
class ACNN(nn.Module):
    def __init__(self, num_classes= 50):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,7,stride=2,padding=3,bias=False) , nn.BatchNorm2d(64), nn.ReLU(inplace=True) , nn.MaxPool2d(3,stride=2,padding=1))
        self.layer1 = nn.ModuleList([ResBk(64,64) for i in range(3) ])
        self.layer2 = nn.ModuleList([ResBk(64 if i== 0 else 128 , 128) for i in range(4) ])
        self.layer3 = nn.ModuleList([ResBk(128 if i== 0 else 256 , 256) for i in range(6) ])
        self.layer4 = nn.ModuleList([ResBk(256 if i== 0 else 512 , 512) for i in range(3) ])
        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.dp = nn.Dropout(.5)
        self.fc = nn.Linear(512,num_classes)
    def forward(self ,x):
        x= self.conv1(x)
        for block in self.layer1:
            x=block(x)
        for block in self.layer2:
            x=block(x)
        for block in self.layer3:
            x=block(x)
        for block in self.layer4:
            x=block(x)
        x = self.avgpool1(x)
        x = x.view(x.size(0),-1)
        x = self.dp(x)
        x = self.fc(x)
        return x


        