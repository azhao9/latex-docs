import torch
import torch.nn as nn
import torchvision.datasets
import torch.nn.functional as F
from torch.autograd import Variable

##TO-DO: Import data here:




##


##TO-DO: Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden1 = nn.Linear(784, 128)
        self.norm1 = nn.BatchNorm1d(128)
        self.hidden2 = nn.Linear(128, 64)
        self.norm2 = nn.BatchNorm1d(64)
        self.hidden3 = nn.Linear(64, 10)
        self.norm3 = nn.BatchNorm1d(10)
            
    def forward(self, x):

        ##Define how forward pass / inference is done:

        x = x.view(-1, 784)
        
        x = self.norm1(self.hidden1(x))
        x = F.relu(x)
        x = self.norm2(self.hidden2(x))
        x = F.relu(x)
        x = self.norm3(self.hidden3(x))
        x = F.log_softmax(x, dim=1)
        
        return x
        
        #return out #return output

my_net = Net()


