import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

##TO-DO: Import data here:

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='.', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='.', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
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

##TO-DO: Train your model:

criterion = nn.NLLLoss()
optimizer = optim.SGD(my_net.parameters(), lr=0.1)

for epoch in range(1):  # loop over the dataset multiple times
    correct = 0
    for i_batch, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = my_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

torch.save(my_net.state_dict(), 'model.pkl')
