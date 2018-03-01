
# Import standard Python packages
import math
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
import copy

import torchvision
import torchvision.transforms as transforms



### Function definitions ###
def addOneToAllWeights(m):
  if hasattr(m, 'weight'):
    m.weight.data += 1


def crossEntropyLoss( outputs, labels ):
  """
  Compute the cross entropy loss given outputs and labels.

  Inputs:
    outputs: (Variable) dimension batch_size x 6 - output of the model
    labels: (Variable) dimension batch_size, where each element is a value in [0, 1, ..., nClasses]

  Output:
    loss (Variable): cross entropy loss for all images in the batch
  """
  num_examples = outputs.size()[0]
  return -torch.sum(outputs[range(num_examples), labels])/num_examples


def findNumParameters( net ):
  nParameters = 0
  for p in net.parameters():
    nParameters += np.prod( list(p.size()) )
  return nParameters


def imshow(img):  # function to show an image
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow( np.transpose( npimg, (1, 2, 0) ) )


def loadData( datacase=0 ):

  if datacase == 0:
    dataDir = '/Volumes/NDWORK128GB/cs230Data/cifar10'
    #dataDir = '/Volumes/Seagate2TB/Data/cifar10'

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


def multi_getattr(obj, attr, default = None):
  """
  Get a named attribute from an object; multi_getattr(x, 'a.b.c.d') is
  equivalent to x.a.b.c.d. When a default argument is given, it is
  returned when any attribute in the chain doesn't exist; without
  it, an exception is raised when a missing attribute is encountered.

  """
  attributes = attr.split(".")
  for i in attributes:
    try:
      obj = getattr(obj, i)
    except AttributeError:
      if default:
        return default
      else:
        raise
  return obj


def multi_setattr(obj, attr, value, default = None):
  """
  Get a named attribute from an object; multi_getattr(x, 'a.b.c.d') is
  equivalent to x.a.b.c.d. When a default argument is given, it is
  returned when any attribute in the chain doesn't exist; without
  it, an exception is raised when a missing attribute is encountered.

  """
  attributes = attr.split(".")
  for i in attributes:
    try:
      if i == attributes[-1]:
        setattr(obj, i, value)
      else:
        obj = getattr(obj, i)

    except AttributeError:
      if default:
        return default
      else:
        raise


def printLayerNames(net):
  for (name,layer) in net._modules.items():
    print(name)  # prints the names of all the parameters


def proxL2L1(m,t):
  if hasattr(m, 'weight'):
    normData = np.sqrt( torch.sum( torch.mul( m.weight.data, m.weight.data ) ) )
    if normData > t:
      m.weight.data = m.weight.data - torch.mul( m.weight.data, t/normData )
    else:
      m.weight.data[:] = 0


def proxL2Lhalf(m,t):
  if hasattr( m, 'weight' ):
    normWeights = np.sqrt( torch.sum( torch.mul( m.weight.data, m.weight.data ) ) )
    if normWeights == 0:
      m.weight.data[:] = 0
    else :
      alpha = t / np.power( normWeights, 1.5 )
      if alpha < 2*np.sqrt(6)/9:
        s = 2 / np.sqrt(3) * np.sin( 1/3 * np.arccos( 3 * np.sqrt(3)/4 * alpha ) + math.pi/2 )
        m.weight.data = (s*s) * m.weight.data
      else:
        m.weight.data[:] = 0

    normWeights = np.sqrt(torch.sum(torch.mul(m.weight.data, m.weight.data)))


def softThreshWeights(m,t):
  # Apply a soft threshold with parameter t to the weights of a nn.Module object
  if hasattr(m, 'weight'):
    m.weight.data = torch.sign(m.weight.data) * torch.clamp( torch.abs(m.weight.data) - t, min=0 )


def trainWithStochFISTA_regL1Norm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL1

  nParameters = findNumParameters( net )

  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      lastParams = {k:v.clone() for k,v in net.state_dict().items()}
      optimizer.zero_grad()

      # Perform a gradient update
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Apply soft threshold to all weights
      net.apply( lambda w: softThreshWeights( w, t=regParam/nParameters ) )

      # determine the cost for the current parameters
      mainLoss = criterion( outputs, labels )
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      # Perform the acceleration update
      theseParams = {k:v.clone() for k,v in net.state_dict().items()}
      for paramName, paramValue in theseParams.items():
        paramValueArray = paramValue.numpy()
        lastParamValue = lastParams[ paramName ]
        lastParamValueArray = lastParamValue.numpy()
        newArray = paramValueArray + k/(k+3) * ( paramValueArray - lastParamValueArray )
        theseParams[ paramName ] = torch.from_numpy( newArray )

      net.load_state_dict( theseParams )

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochProxGradDescent_regL1Norm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL1

  nParameters = findNumParameters( net )

  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)
      optimizer.zero_grad()

      # Perform a gradient update
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Perform a proximal operator update
      net.apply( lambda w: softThreshWeights(w,t=regParam/nParameters) )

      # Determine the current objective function's value
      mainLoss = criterion(outputs, labels)
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochProxGradDescent_regL21Norm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL1

  nParameters = findNumParameters( net )

  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)
      optimizer.zero_grad()

      # Perform a gradient descent update
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      costs[k] = loss.data[0]
      optimizer.step()

      # Perform a proximal operator update
      net.apply( lambda w: proxL2L1(w,t=regParam/nParameters) )

      # Determine the current objective function's value
      mainLoss = criterion(outputs, labels)
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + torch.sqrt( W.norm(2) )
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochProxGradDescent_regL2LhalfNorm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL2Lhalf

  nParameters = findNumParameters( net )

  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)
      optimizer.zero_grad()

      # Perform a gradient descent update
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # Perform a proximal update
      net.apply( lambda w: proxL2Lhalf(w,t=regParam/nParameters) )

      # Determine the current objective function's value
      mainLoss = criterion(outputs, labels)
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + torch.sqrt( W.norm(2) )
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochProxGradDescentwLS_regL1Norm( net, criterion, params ):
  nEpochs = params.nEpochs
  regParam = params.regParam_normL1
  s = params.s
  r = params.r

  nParameters = findNumParameters( net )

  lastT = 1
  optimizer = optim.SGD( net.parameters(), lr=lastT )

  k = 0
  costs = [None] * (nEpochs * len(trainloader))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)
      optimizer.zero_grad()

      # Evaluate the unregularized loss function and calculate the gradient before the update
      preOutputs = net( inputs )
      preLoss = criterion( preOutputs, labels )
      preLoss.backward()

      # Store the result
      preNet = copy.deepcopy( net )
      for paramName, paramValue in net.named_parameters():
        if hasattr( paramValue, 'grad' ):
          thisGrad = multi_getattr( net, paramName+'.grad' )
          multi_setattr( preNet, paramName+'.grad', thisGrad )

      t = s * lastT
      while True:
        # Copy the network's parameters and gradient value
        net = copy.deepcopy( preNet )
        for paramName, preParamValue in preNet.named_parameters():
          if hasattr( preParamValue, 'grad' ):
            thisGrad = multi_getattr( preNet, paramName+'.grad' )
            multi_setattr( net, paramName+'.grad', thisGrad )

        # Perform a gradient descent update
        for param_group in optimizer.param_groups:
          param_group['lr'] = t
        optimizer.step()

        # Perform a proximal operator update
        net.apply( lambda w: softThreshWeights( w, t=t*regParam/nParameters ) )

        # Evaluate the unregularized loss function after the update
        postOutputs = net( inputs )
        postLoss = criterion( postOutputs, labels )

        # Determine if the line search has been satisfied
        normL2Loss = 0
        dpLoss = 0
        for paramName, paramValue in net.named_parameters():
          preGradArray = multi_getattr( preNet, paramName+'.grad.data' ).numpy()
          preValueArray = multi_getattr( preNet, paramName+'.data' ).numpy()
          postValueArray = paramValue.data.numpy()
          diffValueArray = postValueArray - preValueArray
          dpLoss += np.sum( preGradArray * diffValueArray )
          normL2Loss += np.sum( np.square( diffValueArray ) )
        normL2Loss = normL2Loss / (2*t)

        if postLoss.data[0] <= preLoss.data[0] + dpLoss + normL2Loss:
          lastT = t
          break
        t *= r

      mainLoss = criterion(outputs, labels)
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]
      #costs[k] = preLoss.data[0]
      #for paramName, paramValue in net.named_parameters():
      #  costs[k] += torch.sum( torch.abs( paramValue ) ).data[0]

      running_loss += postLoss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochSubGradDescent( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate

  #optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=0.9 )
  optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

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
      costs[k] = loss.data[0]
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochSubGradDescent_regL1Norm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL1

  nParameters = findNumParameters( net )

  #optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=0.9 )
  optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range( nEpochs ):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      optimizer.zero_grad()

      outputs = net(inputs)

      mainLoss = criterion( outputs, labels )

      regLoss = Variable( torch.FloatTensor(1), requires_grad=True)
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      loss.backward()
      optimizer.step()

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochSubGradDescent_regL21Norm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL21

  nParameters = findNumParameters( net )

  #optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=0.9 )
  optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range( nEpochs ):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      optimizer.zero_grad()

      outputs = net(inputs)

      mainLoss = criterion( outputs, labels )

      regLoss = Variable( torch.FloatTensor(1), requires_grad=True )
      for W in net.parameters():
        regLoss = regLoss + W.norm(2)
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      loss.backward()
      optimizer.step()

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs


def trainWithStochSubGradDescent_regL2LhalfNorm( net, criterion, params ):
  nEpochs = params.nEpochs
  learningRate = params.learningRate
  regParam = params.regParam_normL2Lhalf

  nParameters = findNumParameters( net )

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=0.9 )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * len(trainloader) )
  for epoch in range( nEpochs ):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)
      optimizer.zero_grad()

      outputs = net(inputs)

      mainLoss = criterion( outputs, labels )

      regLoss = Variable( torch.FloatTensor(1), requires_grad=True )
      for W in net.parameters():
        regLoss = regLoss + torch.sqrt( W.norm(2) )
      loss = mainLoss + torch.mul( regLoss, regParam/nParameters )
      costs[k] = loss.data[0]

      loss.backward()
      optimizer.step()

      running_loss += loss.data[0]
      if i % 1000 == 999:    # print every 1000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
        running_loss = 0.0

      k += 1

  return costs



### Object definitions ###

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d( 3, 6, 5 )
    self.conv2 = nn.Conv2d( 6, 16, 5 )
    self.fc1 = nn.Linear( 16 * 5 * 5, 120 )
    self.fc2 = nn.Linear( 120, 84 )
    self.fc3 = nn.Linear( 84, 10 )

  def forward(self, x):
    x = F.avg_pool2d( F.softplus( self.conv1(x), beta=100 ), 2, 2 )
    x = F.avg_pool2d( F.softplus( self.conv2(x), beta=100 ), 2, 2 )
    x = x.view( -1, 16 * 5 * 5 )  # converts matrix to vector
    x = F.relu( self.fc1(x) )
    x = F.relu( self.fc2(x) )
    x = self.fc3( x )
    x = F.log_softmax( x, dim=1 )
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



# Parameters for this code
class Params:
  datacase = 0
  learningRate = 0.01
  nEpochs = 300
  regParam_normL1 = 0.01
  regParam_normL21 = 0.01
  regParam_normL2Lhalf = 0.01
  seed = 1
  s = 1.25  # Step size scaling parameter (must be greater than 1)
  r = 0.9  # Backtracking line search parameter (must be between 0 and 1)

if __name__ == '__main__':

  params = Params()
  torch.manual_seed( params.seed )

  (trainset, trainloader, testset, testloader, classes) = loadData( params.datacase )

  net = Net()  # this is my model; it has parameters



  # get some random training images
  #dataiter = iter( trainloader )
  #images, labels = dataiter.next()
  #imshow( torchvision.utils.make_grid(images) )  # show images
  #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))  # print labels

  # Tests
  #net.apply(addOneToAllWeights)  # adds one to all of the weights in the model
  #net.apply( lambda w: softThreshWeights(w,t=1) )  #Applies soft threshold to all weights
  #net.apply( lambda w: proxL2L1(w,t=1) )  #Applies soft threshold to all weights

  #list(net.parameters())  # lists the parameter (or weight) values
  #list(net.conv1.parameters())  # lists the parameters of the conv1 layer
  #list(net.conv1.parameters())[0]  # shows the parameters of the conv1 layer
  #printLayerNames( net )  # prints the layer names


  #criterion = nn.CrossEntropyLoss()  # Softmax is embedded in loss function
  criterion = crossEntropyLoss  # Explicit definiton of cross-entropy loss (without softmax)

  # noRegularization
  #trainWithStochSubGradDescent( net, criterion, params )

  # reg L1 Norm
  #costs = trainWithStochSubGradDescent_regL1Norm( net, criterion, params )
  costs = trainWithStochProxGradDescent_regL1Norm( net, criterion, params )
  #costs = trainWithStochFISTA_regL1Norm( net, criterion, params )
  #costs = trainWithStochProxGradDescentwLS_regL1Norm( net, criterion, params )

  # reg L21 Norm
  #costs = trainWithStochSubGradDescent_regL21Norm( net, criterion, params )
  #costs = trainWithStochProxGradDescent_regL21Norm( net, criterion, params )

  # reg L2LhalfNorm
  #costs = trainWithStochSubGradDescent_regL2LhalfNorm( net, criterion, params )
  #costs = trainWithStochProxGradDescent_regL2LhalfNorm( net, criterion, params )


  plt.plot( costs )
  plt.show()

  dataiter = iter(testloader)
  images, labels = dataiter.next()

  # print images
  imshow( torchvision.utils.make_grid(images) )
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


  outputs = net( Variable(images) )
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


