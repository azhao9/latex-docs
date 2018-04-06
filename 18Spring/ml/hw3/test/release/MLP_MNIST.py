import torch
import torch.nn as nn
import torchvision.datasets
from torch.autograd import Variable

##TO-DO: Import data here:




##


##TO-DO: Define your model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ##Define layers making use of torch.nn functions:
    
    def forward(self, x):

        ##Define how forward pass / inference is done:

        
        #return out #return output

my_net = Net()


##TO-DO: Train your model:

#torch.save(my net.state dict(), ’model.pkl’)
