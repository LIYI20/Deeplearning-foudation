
import torch
from torch import nn
from torch.nn import Sequential

MODELRESH=32
MODELRESW=32

# #########################
# TODO: Build your own model here!!!
# #########################
class YourModel(nn.Module):
    def __init__(self, num_classes=100) -> None:
        super().__init__()
        self.layers = []
        # TODO: Build your own layers here!!!
        self.mode1=Sequential(
        nn.Conv2d(3,32,3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(32),
        #[N,32,16,16]
        nn.Conv2d(32,32,3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(32),
        #[N,32,8,8]
        nn.Conv2d(32,64,3,padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(64),
        #[N,64,4,4]
        nn.Conv2d(64,128,3,padding=1),
        nn.ReLU(inplace=True),  
        nn.MaxPool2d(2),
        nn.BatchNorm2d(128),
        #[N,128,2,2]
        nn.Conv2d(128,256,3,padding=1),
        nn.ReLU(inplace=True),
        #[N,256,2,2]
        nn.Flatten(),
        nn.Linear(1024,num_classes),
        nn.Softmax(dim=-1)
        )

    def forward(self,x):
        x=self.mode1(x)
        return x

