
# Import standard Python packages
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re   # Regular expressions

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
  if hasattr(m, 'bias'):
    m.bias.data += 1


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


def determineTrainingError( net, trainloader ):
  # Does not include regularization
  loss = 0
  for j, jData in enumerate( trainloader, 0 ):
    jInputs, jLabels = jData
    jInputs, jLabels = Variable(jInputs), Variable(jLabels)
    jOutputs = net( jInputs )
    thisLoss = criterion( jOutputs, jLabels )
    loss += thisLoss.data[0]
  return loss


def findNumDeadNeurons( net ):
  nDead = 0
  for pName, p in net.named_parameters():
    if re.search( 'weight$', pName ):
      weightData = p.data.numpy()
      nameParts = pName.split('.')
      nameParts[-1] = 'bias'
      bias = multi_getattr(net, ".".join(nameParts) )
      biasData = bias.data.numpy()

      if re.search( '^conv', pName ):
        # Convlutional layer
        nNeurons = p.data.numpy().shape[0]
        for n in range(0, nNeurons):
          thisData = weightData[n, :, :, :]
          thisBias = biasData[n]
          maxData = np.max( np.absolute( thisData ) )
          maxBias = np.max( np.absolute( thisBias ) )
          if np.max([maxData,maxBias]) <= 0:
            nDead += 1
      elif re.search( '^fc', pName ):
        # Fully connected linear layer
        nNeurons = weightData.shape[0]
        for n in range(0, nNeurons):
          maxData = np.max( np.absolute( weightData[:,n] ) )
          maxBias = np.max( biasData[n] )
          if np.max([maxData,maxBias]) <= 0:
            nDead += 1
  return nDead


def findNumWeights( net ):
  nWeights = 0
  for p in net.parameters():
    nWeights += np.prod( list(p.size()) )
  return nWeights


def findTestAccuracy( net, testloader ):
  correct = 0
  total = 0
  for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

  accuracy = correct / total
  return accuracy
  print( 'Accuracy of the network on the 10000 test images: %d' % ( 100 * accuracy ) )


def findNumZeroWeights( net ):
  nZeroParameters = 0
  for p in net.parameters():
    nZeroParameters += p.data.numpy().size - np.count_nonzero( p.data.numpy() )
  return nZeroParameters


def imshow(img):  # function to show an image
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow( np.transpose( npimg, (1, 2, 0) ) )


def loadData( datacase=0, batchSize=100, shuffle=True ):

  if datacase == 0:
    dataDir = '/Volumes/NDWORK128GB/cs230Data/cifar10'
    if not os.path.isdir(dataDir):
      dataDir = '/Volumes/Seagate2TB/Data/cifar10'

    transform = transforms.Compose( [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10( root=dataDir, train=True, download=True, transform=transform )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    testset = torchvision.datasets.CIFAR10( root=dataDir, train=False, download=True, transform=transform )
    testloader = torch.utils.data.DataLoader( testset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainset, trainloader, testset, testloader, classes)

  elif datacase == 1:
    dataDir = '/Volumes/NDWORK128GB/cs230Data/mnist'
    if not os.path.isdir(dataDir):
      dataDir = '/Volumes/Seagate2TB/Data/mnist'

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    trainset = torchvision.datasets.MNIST( root=dataDir, train=True, download=True, transform=transform )
    trainloader = torch.utils.data.DataLoader( trainset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    testset = torchvision.datasets.MNIST( root=dataDir, train=False, download=True, transform=transform )
    testloader = torch.utils.data.DataLoader( testset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return (trainset, trainloader, testset, testloader, classes)

  else:
    print( 'Error: incorrect datacase entered' )


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
    neurWeight = m.weight.data.numpy()
    neurBias = m.bias.data.numpy()
    if isinstance( m, torch.nn.modules.linear.Linear ):
      for n in range(0,len(neurBias)):
        thisData = neurWeight[:,n]
        thisBias = neurBias[n]
        normData = np.sqrt( np.sum(thisData*thisData) + np.sum(thisBias*thisBias) )
        if normData > t:
          neurWeight[:,n] = thisData - thisData * t / normData
          neurBias[n] = thisBias - thisBias * t / normData
        else:
          neurWeight[:,n] = 0
          neurBias[n] = 0
      m.weight.data = torch.from_numpy( neurWeight )
      m.weight.bias = torch.from_numpy( neurBias )

    elif isinstance( m, torch.nn.modules.conv.Conv2d ):
      nNeurons = neurWeight.shape[0]
      for n in range(0,nNeurons):
        thisData = neurWeight[n,:,:,:]
        thisBias = neurBias[n]
        normData = np.sqrt( np.sum(thisData*thisData) + thisBias*thisBias )
        if normData > t:
          m.weight.data[n,:,:,:] = torch.from_numpy( thisData - thisData * t / normData )
          m.bias.data[n] = thisBias - thisBias * t / normData
        else:
          m.weight.data[n,:,:,:] = 0
          m.bias.data[n] = 0


def proxL2LHalf(m,t):
  if hasattr( m, 'weight' ):
    neurWeight = m.weight.data.numpy()
    neurBias = m.bias.data.numpy()
    if isinstance( m, torch.nn.modules.linear.Linear ):
      for n in range(0,len(neurBias)):
        thisData = neurWeight[:,n]
        thisBias = neurBias[n]
        normData = np.sqrt( np.sum(thisData*thisData) + np.sum(thisBias*thisBias) )

        if normData == 0:
          thisData[:] = 0
          thisBias[n] = 0
        else :
          alpha = t / np.power( normData, 1.5 )
          if alpha < 2*np.sqrt(6)/9:
            s = 2 / np.sqrt(3) * np.sin( 1/3 * np.arccos( 3 * np.sqrt(3)/4 * alpha ) + math.pi/2 )
            thisData = (s*s) * thisData 
          else:
            thisData[:] = 0
            thisBias[n] = 0

    elif isinstance( m, torch.nn.modules.conv.Conv2d ):
      for n in range(0,len(neurBias)):
        thisData = neurWeight[n,:,:,:]
        thisBias = neurBias[n]
        normData = np.sqrt( np.sum(thisData*thisData) + np.sum(thisBias*thisBias) )

        if normData == 0:
          thisData[:] = 0
          thisBias[n] = 0
        else :
          alpha = t / np.power( normData, 1.5 )
          if alpha < 2*np.sqrt(6)/9:
            s = 2 / np.sqrt(3) * np.sin( 1/3 * np.arccos( 3 * np.sqrt(3)/4 * alpha ) + math.pi/2 )
            thisData = (s*s) * thisData 
          else:
            thisData[:] = 0
            thisBias[n] = 0

    m.weight.data = torch.from_numpy( thisData )
    m.weight.bias = torch.from_numpy( thisBias )


def showTestResults( net, testloader ):
  # Determine accuracy on test set

  accuracy = findTestAccuracy( net, testloader )
  print( 'Accuracy of the network on the 10000 test images: %d %%' % ( 100 * accuracy ) )

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


def softThreshTwoWeights( m, tConv, tLinear ):
  # Apply a soft threshold to the weights of a nn.Module object
  # Applies one weight to the convolutional layers and another to the fully connected layers
  if isinstance( m, torch.nn.modules.conv.Conv2d ):
    m.weight.data = torch.sign(m.weight.data) * torch.clamp( torch.abs(m.weight.data) - tConv, min=0 )
    m.bias.data = torch.sign(m.bias.data) * torch.clamp( torch.abs(m.bias.data) - tConv, min=0 )
  elif isinstance( m, torch.nn.modules.linear.Linear ):
    m.weight.data = torch.sign(m.weight.data) * torch.clamp( torch.abs(m.weight.data) - tLinear, min=0 )
    m.bias.data = torch.sign(m.bias.data) * torch.clamp( torch.abs(m.bias.data) - tLinear, min=0 )


def softThreshWeights(m,t):
  # Apply a soft threshold with parameter t to the weights of a nn.Module object
  if hasattr(m, 'weight'):
    m.weight.data = torch.sign(m.weight.data) * torch.clamp( torch.abs(m.weight.data) - t, min=0 )
  if hasattr(m, 'bias'):
    m.bias.data = torch.sign(m.bias.data) * torch.clamp( torch.abs(m.bias.data) - t, min=0 )


def trainWithProxGradDescent_regL1Norm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL1

  optimizer = optim.SGD( net.parameters(), lr=learningRate )
  nWeights = findNumWeights( net )

  k = 0
  costs = [None] * nEpochs
  sparses = [None] * nEpochs
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    optimizer.zero_grad()
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient
      outputs = net( inputs )
      thisLoss = criterion( outputs, labels )
      thisLoss.backward()

      if i >= nBatches-1:
        break

    # Perform a gradient descent update
    optimizer.step()

    # Perform a proximal operator update
    net.apply( lambda w: softThreshWeights( w, t=learningRate*regParam/nWeights ) )

    # Determine the current objective function's value
    mainLoss = criterion( outputs, labels )
    regLoss = 0
    for W in net.parameters():
      regLoss = regLoss + W.norm(1)
    regLoss = torch.mul( regLoss, regParam/nWeights )
    loss = mainLoss + regLoss
    costs[epoch] = loss.data[0]
    sparses[epoch] = findNumZeroWeights( net )

    if epoch % printEvery == printEvery-1:
      print( '[%d] cost: %.3f' % (epoch+1, costs[k] ) )

    k += 1

  return ( costs, sparses )


def trainWithProxGradDescent_regL2L1Norm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL2L1

  optimizer = optim.SGD( net.parameters(), lr=learningRate )
  nWeights = findNumWeights( net )

  k = 0
  costs = [None] * nEpochs
  groupSparses = [None] * nEpochs
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    optimizer.zero_grad()
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient
      outputs = net( inputs )
      thisLoss = criterion( outputs, labels )
      thisLoss.backward()

      if i >= nBatches-1:
        break

    # Perform a gradient descent update
    optimizer.step()

    # Perform a proximal operator update
    net.apply( lambda w: proxL2L1( w, t=learningRate*regParam/nWeights ) )

    # Determine the current objective function's value
    mainLoss = criterion( outputs, labels )
    regLoss = 0
    for W in net.parameters():
      regLoss = regLoss + W.norm(2)
    regLoss = torch.mul( regLoss, regParam/nWeights )
    loss = mainLoss + regLoss
    costs[epoch] = loss.data[0]

    groupSparses[epoch] = findNumDeadNeurons( net )

    if epoch % printEvery == printEvery-1:
      print( '[%d] cost: %.3f' % (epoch+1, costs[k] ) )

    k += 1

  return ( costs, groupSparses )


def trainWithStochProxGradDescent_regL1Norm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL1

  nWeights = findNumWeights( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  sparses = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      costs[k] = loss.data[0]

      # Perform a gradient descent update
      optimizer.step()

      # Perform a proximal operator update
      net.apply( lambda w: softThreshWeights( w, t=learningRate*regParam/nWeights ) )

      # Determine the current objective function's value
      mainLoss = 0
      for j, jData in enumerate( trainloader, 0 ):
        jInputs, jLabels = jData
        jInputs, jLabels = Variable(jInputs), Variable(jLabels)
        jOutputs = net( jInputs )
        thisLoss = criterion( jOutputs, jLabels )
        mainLoss += thisLoss.data[0]
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      regLoss = torch.mul( regLoss, regParam/nWeights )
      loss = mainLoss + regLoss.data[0]
      costs[k] = loss
      sparses[k] = findNumZeroWeights( net )

      if k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f' % ( epoch+1, i+1, costs[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, sparses )


def trainWithStochProxGradDescent_regL2L1Norm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL2L1

  nWeights = findNumWeights( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  groupSparses = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net( inputs )
      loss = criterion( outputs, labels )
      optimizer.zero_grad()
      loss.backward()

      # Perform a gradient descent update
      optimizer.step()

      # Perform a proximal operator update
      net.apply( lambda w: proxL2L1( w, t=learningRate*regParam/nWeights ) )

      # Determine the current objective function's value
      mainLoss = criterion( outputs, labels )
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(2)
      regLoss = torch.mul( regLoss, regParam/nWeights )
      loss = mainLoss + regLoss
      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )

      if k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f,  group sparsity: %d' % \
          ( epoch+1, i+1, costs[k], groupSparses[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithStochProxGradDescent_regL2LHalfNorm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL2L1

  nWeights = findNumWeights( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  groupSparses = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net( inputs )
      loss = criterion( outputs, labels )
      optimizer.zero_grad()
      loss.backward()

      # Perform a gradient descent update
      optimizer.step()

      # Perform a proximal operator update
      net.apply( lambda w: proxL2LHalf( w, t=learningRate*regParam/nWeights ) )

      # Determine the current objective function's value
      mainLoss = criterion( outputs, labels )
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(2)
      regLoss = torch.mul( regLoss, regParam/nWeights )
      loss = mainLoss + regLoss
      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )

      if k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f,  group sparsity: %d' % \
          ( epoch+1, i+1, costs[k], groupSparses[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithStochSubGradDescent( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()
      costs[k] = loss.data[0]

      optimizer.step()

      if k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f' % ( epoch+1, i+1, costs[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return costs


def trainWithStochSubGradDescent_regL1Norm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL1

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  nWeights = findNumWeights(net)

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  sparses = [None] * (nEpochs * np.min([len(trainloader), nBatches]))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      mainLoss = criterion( outputs, labels )
      regLoss = Variable( torch.FloatTensor(1), requires_grad=True)
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      loss = mainLoss + torch.mul( regLoss, regParam/nWeights )
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0]
      sparses[k] = findNumZeroWeights(net)

      optimizer.step()

      if k % printEvery == printEvery-1:
        print( '[%d,%d] cost: %.3f' % ( epoch+1, k+1, costs[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return (costs,sparses)


def trainWithStochSubGradDescent_regL2L1Norm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL2L1

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  nWeights = findNumWeights(net)

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  groupSparses = [None] * (nEpochs * np.min([len(trainloader), nBatches]))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      mainLoss = criterion( outputs, labels )
      regLoss = Variable( torch.FloatTensor(1), requires_grad=True )
      for thisMod in net.modules():
        print( "I got here" )
      for pName, pValue in net.named_parameters():
        # TODO: Use isinstance like shown below
        #if isinstance( m, torch.nn.modules.conv.Conv2d ):
        #elif isinstance( m, torch.nn.modules.linear.Linear ):
        # Instead of iterating over parameters() or named_parameters(), could iterate over
        # modules() instead
        nameParts = pName.split('.')
        nameParts[-1] = 'bias'
        bias = multi_getattr(net, ".".join(nameParts) )

        if re.search( '^conv.weight', pName ):
          nNeurons = pValue.shape[0]
          for n in range(0,nNeurons):
            regLoss = regLoss + torch.sqrt( torch.mul( pValue[n,:,:,:].norm(2), pValue[n,:,:,:].norm(2) ) + \
              torch.mul( bias[n], bias[n] ) )
        elif re.search( '^fc.weight', pName ):
          nNeurons = pValue.shape[0]
          for n in range(0,nNeurons):
            regLoss = regLoss + torch.sqrt( torch.mul( pValue[n,:].norm(2), pValue[n,:].norm(2) ) + \
              torch.mul( bias[n], bias[n] ) )

      loss = mainLoss + torch.mul( regLoss, regParam/nWeights )
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )

      optimizer.step()

      if k % printEvery == printEvery-1:
        print( '[%d,%d] cost: %.3f' % ( epoch+1, k+1, costs[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithStochSubGradDescent_regL2LHalfNorm( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL2L1

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  nWeights = findNumWeights(net)

  k = 0
  costs = [None] * ( nEpochs * np.min([len(trainloader),nBatches]) )
  groupSparses = [None] * (nEpochs * np.min([len(trainloader), nBatches]))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      mainLoss = criterion( outputs, labels )
      regLoss = Variable( torch.FloatTensor(1), requires_grad=True )
      for W in net.parameters():
        regLoss = regLoss + torch.sqrt( W.norm(2) )
      loss = mainLoss + torch.mul( regLoss, regParam/nWeights )
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )

      optimizer.step()

      if k % printEvery == printEvery-1:
        print( '[%d,%d] cost: %.3f' % ( epoch+1, k+1, costs[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithSubGradDescent( net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * nEpochs
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    optimizer.zero_grad()
    loss = 0
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient
      outputs = net(inputs)
      thisLoss = criterion(outputs, labels)
      thisLoss.backward()
      loss += thisLoss.data[0]

      if i >= nBatches-1:
        break

    optimizer.step()
    costs[epoch] = loss

    if epoch % printEvery == printEvery-1:
      print( '[%d] cost: %.3f' % (epoch+1, costs[k] ) )

    k += 1

  return costs


def trainWithSubGradDescentLS( net, criterion, params ):
  alpha = params.alpha
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  s = params.s
  r = params.r

  lastT = 1
  costs = [None] * nEpochs
  k = 0
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    optimizer = optim.SGD( net.parameters(), lr=lastT )

    # Evaluate the loss function before the gradient descent step
    preLoss = 0
    optimizer.zero_grad()
    for i, data in enumerate( trainloader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient
      outputs = net( inputs )
      tmpLoss = criterion( outputs, labels )
      tmpLoss.backward()
      preLoss += tmpLoss.data[0]

      if i >= nBatches-1:
        break

    costs[epoch] = preLoss

    # Average summed values for gradient
    for paramName, paramValue in net.named_parameters():
      if hasattr( paramValue, 'grad' ):
        thisGrad = multi_getattr( net, paramName+'.grad' )
        #multi_setattr( net, paramName+'.grad', thisGrad/len(trainloader) )

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
      optimizer = optim.SGD( net.parameters(), lr=t )
      optimizer.step()

      # Evaluate the loss function after the update
      postLoss = 0
      for i, data in enumerate( trainloader, 0 ):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # Calculate the gradient
        outputs = net( inputs )
        tmpLoss = criterion( outputs, labels )
        postLoss += tmpLoss.data[0]

        if i >= nBatches-1:
          break

      # Evaluate the dot product component of the line search criteria
      dpLoss = 0
      for paramName, paramValue in net.named_parameters():
        preGradArray = multi_getattr( preNet, paramName+'.grad.data' ).numpy()
        preValueArray = multi_getattr( preNet, paramName+'.data' ).numpy()
        postValueArray = paramValue.data.numpy()
        diffValueArray = postValueArray - preValueArray
        thisDpLoss = np.sum( preGradArray * diffValueArray )
        if thisDpLoss > 0:
          print("I got here")
        dpLoss += thisDpLoss

      # Determine if the line search has been satisfied
      if postLoss <= preLoss + alpha * t * dpLoss:
        lastT = t
        break
      t *= r

    if postLoss > preLoss:
      print("I got here")

    # if epoch % 10 == 9:
    print( '[%d] cost: %.3f' % (epoch+1, costs[k] ) )

    k += 1

  return costs



### Object definitions ###

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d( 3, 30, 3 )  # inChannels, outChannels, kSize
    self.conv2 = nn.Conv2d( 30, 30, 3 )
    self.conv3 = nn.Conv2d( 30, 20, 3 )
    self.fc1 = nn.Linear( 20 * 6 * 6, 120 )
    self.fc2 = nn.Linear( 120, 80 )
    self.fc3 = nn.Linear( 80, 10 )

  def forward(self, x):
    x = F.softplus( self.conv1(x), beta=100 )
    x = F.avg_pool2d( F.softplus( self.conv2(x), beta=100 ), 2, 2 )
    x = F.avg_pool2d( F.softplus( self.conv3(x), beta=100 ), 2, 2 )
    x = x.view( -1, 20 * 6 * 6 )  # converts matrix to vector
    x = F.softplus( self.fc1(x), beta=100 )
    x = F.softplus( self.fc2(x), beta=100 )
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
  batchSize = 500
  datacase = 0
  momentum = 0.0
  nBatches = 1000000
  nEpochs = 100
  printEvery = 1
  regParam_normL1 = 1e2
  regParam_normL2L1 = 1e2
  regParam_normL2Lhalf = 1e2
  seed = 1
  shuffle = False  # Shuffle the data in each minibatch
  alpha = 0.8
  s = 1.25  # Step size scaling parameter (must be greater than 1)
  r = 0.9  # Backtracking line search parameter (must be between 0 and 1)


if __name__ == '__main__':

  params = Params()
  torch.manual_seed( params.seed )

  (trainset, trainloader, testset, testloader, classes) = loadData( \
    params.datacase, params.batchSize, params.shuffle )

  net = Net()  # this is my model; it has parameters



  # get some random training images
  #dataiter = iter( trainloader )
  #images, labels = dataiter.next()
  #imshow( torchvision.utils.make_grid(images) )  # show images
  #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))  # print labels

  # Tests
  #net.apply(addOneToAllWeights)  # adds one to all of the weights in the model
  #net.apply( lambda w: softThreshWeights(w,t=1) )  #Applies soft threshold to all weights
  #net.apply( lambda w: softThreshTwoWeights(w,tConv=1,tLinear=10) )  #Applies two soft thresholds to weights
  #net.apply( lambda w: proxL2L1(w,t=1) )  #Applies soft threshold to all weights

  #list(net.parameters())  # lists the parameter (or weight) values
  #list(net.conv1.parameters())  # lists the parameters of the conv1 layer
  #list(net.conv1.parameters())[0]  # shows the parameters of the conv1 layer
  #printLayerNames( net )  # prints the layer names


  #criterion = nn.CrossEntropyLoss()  # Softmax is embedded in loss function
  criterion = crossEntropyLoss  # Explicit definiton of cross-entropy loss (without softmax)

  # noRegularization
  #costs = trainWithSubGradDescent( net, criterion, params, learningRate=0.1 )
  #costs = trainWithSubGradDescentLS( net, criterion, params, learningRate=0.1 )
  #costs = trainWithStochSubGradDescent( net, criterion, params, learningRate=0.1 )

  # L1 norm regularization
  #(costs,sparses) = trainWithStochSubGradDescent_regL1Norm( net, criterion, params, learningRate=0.1 )
  #(costs,sparses) = trainWithStochProxGradDescent_regL1Norm( net, criterion, params, learningRate=0.1 )
  #(costs, sparses) = trainWithProxGradDescent_regL1Norm(net, criterion, params, learningRate=0.1 )

  # L2,L1 norm regularization
  #(costs,groupSparses) = trainWithProxGradDescent_regL2L1Norm( net, criterion, params, learningRate=0.1 )
  #(costs,groupSparses) = trainWithStochSubGradDescent_regL2L1Norm( net, criterion, params, learningRate=0.1 )
  #(costs,groupSparses) = trainWithStochProxGradDescent_regL2L1Norm( net, criterion, params, learningRate=0.1 )

  #L2,L1/2 norm regularization
  #(costs,groupSparses) = trainWithStochSubGradDescent_regL2LHalfNorm( net, criterion, params, learningRate=0.1 )
  #(costs,groupSparses) = trainWithStochProxGradDescent_regL2LHalfNorm( net, criterion, params, learningRate=0.1 )


  # Experiment to determine the best learning rate for L2L1 regularization
  (costs,groupSparses) = trainWithStochSubGradDescent_regL2L1Norm( net, criterion, params, learningRate=0.1 )

  testAccuracy = findTestAccuracy( net, testloader )

  with open('trainWithStochProxGradDescent_regL2L1Norm_0pt1.pkl', 'wb') as f:
    pickle.dump( (params, net, testAccuracy, costs0pt1, groupSparses0pt1), f)

  plt.plot( costs0pt1 )
  plt.title('Stochastic Proximal Gradient with L2,L1 Regularization 0.1')
  plt.show()

  plt.plot( groupSparses0pt1 )
  plt.title('Stochastic Proximal Gradient with L2,L1 Regularization 0.1')
  plt.show()



  showTestResults( net, testloader )

  dataiter = iter(testloader)
  images, labels = dataiter.next()

  # print images
  imshow( torchvision.utils.make_grid(images) )
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


  outputs = net( Variable(images) )
  _, predicted = torch.max( outputs.data, 1 )

  print( 'Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)) )


