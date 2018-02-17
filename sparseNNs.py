
# Import standard Python packages
import matplotlib.pyplot as plt
import numpy as np

# Import torch and torchvision packages
import torch
import torch.autograd as autograd
from   torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# Parameters for this code
datacase = 0
torch.manual_seed(1)


# Function definitions

def loadData( datacase=0 ):

  if datacase == 0:
    dataDir = '/Volumes/NDWORK128GB/cs230Data/cifar10'

    transform = transforms.Compose( [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10( root=dataDir, train=True, download=True, transform=transform )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=4, shuffle=True, num_workers=2 )

    testset = torchvision.datasets.CIFAR10( root=dataDir, train=False, download=True, transform=transform )
    testloader = torch.utils.data.DataLoader( testset, batch_size=4, shuffle=False, num_workers=2 )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainset, trainloader, testset, testloader, classes)

  else:
    print( 'Error: incorrect datase entered' )


def imshow(img):  # function to show an image
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow( np.transpose( npimg, (1, 2, 0) ) )


# Object definitions

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool  = nn.MaxPool2d(2, 2)  # parameter free
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    #x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # equivalent to line above
      # this makes it so only the layers withparameters are in the init function
    x = self.pool(F.relu(self.conv2(x)))
    #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 16 * 5 * 5)  # converts matrix to vector
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x



### Main Code ###

# Questions for Surag:
# How can I change the weights (e.g. with softthresh)?  https://discuss.pytorch.org/t/how-to-modify-weights-of-layers-in-resnet/2867
# How do I change the network to have X layers with Y nodes in each layer?
# How do I include a softmax as the final layer?  What if I wanted just a linear output (for all real numbers)?
# What are trainloader, testloader?  How can I use other datasets?
  # Look at "Classifying Images of Hand Signs" to make dataset and dataloader objects
# Why doesn't the example show the CIFAR10 images when running using PyCharm (only shows in debug mode)?
# Why doesn't the example show images when running from the command line?




# plt.imshow( np.zeros((100,100)) )  # This works fine


(trainset, trainloader, testset, testloader, classes) = loadData( datacase )

# get some random training images
dataiter = iter( trainloader )
images, labels = dataiter.next()

# show images
imshow( torchvision.utils.make_grid(images) )
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))





net = Net()  # this is my model; it has parameters


#def addOne(m):
#  if type(m) == nn.Conv2d:
#    print(m)
#    m.weight.data += 1
#
#net.apply(addOne)   # adds one to all of the weights of the convolution layers
#
#def addOneToAllWeights(m):
#  if hasattr(m, 'weight'):
#    m.weight.data += 1
#
#net.apply(addOneToAllWeights)  # adds one to all of the weights in the model
#
#def softThreshWeights(m,t):
#  if hasattr(m, 'weight'):
#    m.weight.data = torch.sign(m.weight.data) * torch.max( ( torch.abs(m.weight.data) - t ), 0 ) 
#
## net.apply(addOneToAllWeights)
##net.apply( lambda w: softThreshWeights(w,t=1) )


#def printLayerNames(net):
#  for (name,layer) in net._modules.items():
#    print(name)  # prints the names of all the parameters

#list(net.parameters())  # lists the parameter values
#list(net.conv1.parameters())[0]  # shows the parameters of the conv1 layer


#http://pytorch.org/docs/master/nn.html

#list(net.conv1.parameters())  # lists the weights of the conv1 layer


criterion = nn.CrossEntropyLoss()
  # Softmax is embedded in loss function
  # look at net.py (by Surag) to see how to do it explicitly
optimizer = optim.SGD( net.parameters(), lr=0.001, momentum=0.9 )

for epoch in range(2):  # loop over the dataset multiple times

  running_loss = 0.0
  for i, data in enumerate( trainloader, 0 ):
    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.data[0]
    if i % 1000 == 999:    # print every 1000 mini-batches
      print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
      running_loss = 0.0

print('Finished Training')



dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)

print( 'Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)) )


# Determine accuracy on test set
correct = 0
total = 0
for data in testloader:
  images, labels = data
  outputs = net(Variable(images))
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
  images, labels = data
  outputs = net(Variable(images))
  _, predicted = torch.max(outputs.data, 1)
  c = (predicted == labels).squeeze()
  for i in range(4):
    label = labels[i]
    class_correct[label] += c[i]
    class_total[label] += 1


for i in range(10):
  print('Accuracy of %5s : %2d %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))


