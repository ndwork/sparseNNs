
# Import standard Python packages
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
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
  return -torch.sum(outputs[range(num_examples), labels])


def determineError( net, dataLoader ):
  # Does not include regularization
  loss = 0
  for j, jData in enumerate( dataLoader, 0 ):
    jInputs, jLabels = jData
    jInputs, jLabels = Variable(jInputs), Variable(jLabels)
    jOutputs = net( jInputs )
    thisLoss = criterion( jOutputs, jLabels )
    loss += thisLoss.data[0]
  return loss


def findNumDeadNeurons( net, thresh=0 ):
  nDead = 0
  for thisMod in net.modules():
    if hasattr(thisMod, 'weight'):
      weightData = thisMod.weight.data.cpu().numpy()
      biasData = thisMod.bias.data.cpu().numpy()

      if isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        nNeurons = weightData.shape[0]
        for n in range(0, nNeurons):
          thisWeight = weightData[n,:,:,:]
          thisBias = biasData[n]
          thisMag = np.sqrt( np.sum( thisWeight*thisWeight) + thisBias*thisBias )
          if thisMag <= thresh: nDead += 1

      elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
          nNeurons = weightData.shape[0]
          for n in range(0, nNeurons):
            thisWeight = weightData[n,:]
            thisBias = biasData[n]
            thisMag = np.sqrt( np.sum( thisWeight*thisWeight) + thisBias*thisBias )
            if thisMag <= thresh: nDead += 1

  return nDead


def findNumLiveNeuronsInLayers( net, thresh=0 ):
  nLayers = len( net._modules.items() )
  nLives = [None] * nLayers
  for thisName, thisMod in net._modules.items():
    if hasattr(thisMod, 'weight'): 

      weightData = thisMod.weight.data.cpu().numpy()
      biasData = thisMod.bias.data.cpu().numpy()
  
      thisLive = 0
      if isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        nNeurons = weightData.shape[0]
        for n in range(0, nNeurons):
          thisWeight = weightData[n,:,:,:]
          thisBias = biasData[n]
          thisMag = np.sqrt( np.sum( thisWeight*thisWeight) + thisBias*thisBias )
          if thisMag > thresh: thisLive += 1
  
      elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
        nNeurons = weightData.shape[0]
        for n in range(0, nNeurons):
          thisWeight = weightData[n,:]
          thisBias = biasData[n]
          thisMag = np.sqrt( np.sum( thisWeight*thisWeight) + thisBias*thisBias )
          if thisMag > thresh: thisLive += 1
  
      layerIndx = net.layerNames.index( thisName )
      nLives[ layerIndx ] = thisLive

  return nLives


def findNumNeurons( net ):
  nNeurons = 0
  for thisMod in net.modules():
    if hasattr(thisMod, 'weight'):
      weightData = thisMod.weight.data.cpu().numpy()
      if isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        nNeurons += weightData.shape[0]
      elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
        nNeurons += weightData.shape[0]

  return nNeurons


def findNumWeights( net ):
  nWeights = 0
  for p in net.parameters():
    nWeights += np.prod( list(p.size()) )
  return nWeights


def findAccuracy( net, dataLoader, cuda ):
  correct = 0
  total = 0
  for data in dataLoader:
    inputs, labels = data
    if cuda:
      inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
    outputs = net( Variable(inputs) )

    _, predicted = torch.max( outputs.data, 1 )  # max along dimension 1
    total += labels.size(0)
    correct += ( predicted == labels ).sum()

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


def loadCheckpoint( net, checkpointFile, costs=None ):
  net.load_state_dict( torch.load( checkpointFile ) )
  if costs is not None:
    with open( resultFile, 'rb' ) as f:
      testAccuracy, costs, groupSparses = pickle.load(f)


def loadData( datacase=0, batchSize=100, shuffle=True ):

  mainDataDir = '/Volumes/NDWORK128GB/cs230Data/'
  if not os.path.isdir(mainDataDir):
    mainDataDir = '/Volumes/Seagate2TB/Data/'
  if not os.path.isdir(mainDataDir):
    mainDataDir = './data'
  if not os.path.isdir(mainDataDir):
    os.mkdir( mainDataDir )

  if datacase == 0:
    dataDir = mainDataDir + '/cifar10'

    transform = transforms.Compose( [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10( root=dataDir, train=True, download=True, transform=transform )
    trainLoader = torch.utils.data.DataLoader( trainset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    testset = torchvision.datasets.CIFAR10( root=dataDir, train=False, download=True, transform=transform )
    testLoader = torch.utils.data.DataLoader( testset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return (trainset, trainLoader, testset, testLoader, classes)

  elif datacase == 1:
    dataDir = mainDataDir + '/mnist'

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    trainset = torchvision.datasets.MNIST( root=dataDir, train=True, download=True, transform=transform )
    trainLoader = torch.utils.data.DataLoader( trainset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    testset = torchvision.datasets.MNIST( root=dataDir, train=False, download=True, transform=transform )
    testLoader = torch.utils.data.DataLoader( testset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return (trainset, trainLoader, testset, testLoader, classes)

  elif datacase == 2:
    dataDir = mainDataDir = '/stl10'
    if not os.path.isdir(dataDir):
      dataDir = '/Volumes/Seagate2TB/Data/stl10'

    transform = transforms.Compose( [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.STL10( root=dataDir, train=True, download=True, transform=transform )
    trainLoader = torch.utils.data.DataLoader( trainset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    testset = torchvision.datasets.STL10( root=dataDir, train=False, download=True, transform=transform )
    testLoader = torch.utils.data.DataLoader( testset, batch_size=batchSize, shuffle=shuffle, num_workers=2 )

    classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    return (trainset, trainLoader, testset, testLoader, classes)

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


def pruneNet( net, cuda, thresh=0 ):
  nLives = findNumLiveNeuronsInLayers( net, thresh=thresh )
  newNet = smartNet( nLives )

  for thisName, thisMod in net.named_modules():
    if not hasattr(thisMod, 'weight'): continue

    weightData = thisMod.weight.data.cpu().numpy()
    biasData = thisMod.bias.data.cpu().numpy()

    for newName, newMod in newNet.named_modules():
      if newName != thisName: continue

      newWeight = newMod.weight.data.cpu().numpy()
      newBias = newMod.bias.data.cpu().numpy()

      if isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        nNeurons = weightData.shape[0]
        newIndx = 0
        for n in range(0, nNeurons):
          thisWeight = weightData[n,:,:,:]
          thisBias = biasData[n]
          thisMag = np.sqrt( np.sum( thisWeight*thisWeight) + thisBias*thisBias )
          if thisMag > thresh:
            newWeight[newIndx,:,:,:] = thisWeight
            newBias[newIndx] = thisBias
            newIndx += 1

      elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
        nNeurons = weightData.shape[0]
        newIndx = 0
        for n in range(0, nNeurons):
          thisWeight = weightData[n,:]
          thisBias = biasData[n]
          thisMag = np.sqrt( np.sum( thisWeight*thisWeight) + thisBias*thisBias )
          if thisMag > thresh:
            newWeight[newIndx,:] = thisWeight
            newBias[newIndx] = thisBias
            newIndx += 1

      if cuda: 
        newMod.weight.data = torch.from_numpy( newWeight ).cuda()
        newMod.weight.bias = torch.from_numpy( newBias ).cuda()
      else:
        newMod.weight.data = torch.from_numpy( newWeight )
        newMod.weight.bias = torch.from_numpy( newBias )

  net = copy.deepcopy( newNet )
  #return newNet


def proxL2L1( net, t, cuda ):
  for thisMod in net.modules():
    if hasattr( thisMod, 'weight'):
      neurWeight = thisMod.weight.data.cpu().numpy()
      neurBias = thisMod.bias.data.cpu().numpy()
      if isinstance( thisMod, torch.nn.modules.linear.Linear ):
        for n in range(0,len(neurBias)):
          thisData = neurWeight[n,:]
          thisBias = neurBias[n]
          normData = np.sqrt( np.sum(thisData*thisData) + thisBias*thisBias )
          if normData > t:
            neurWeight[n,:] = thisData - thisData * t / normData
            neurBias[n] = thisBias - thisBias * t / normData
          else:
            neurWeight[n,:] = 0
            neurBias[n] = 0

      elif isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        nNeurons = neurWeight.shape[0]
        for n in range(0,nNeurons):
          thisData = neurWeight[n,:,:,:]
          thisBias = neurBias[n]
          normData = np.sqrt( np.sum(thisData*thisData) + thisBias*thisBias )
          if normData > t:
            neurWeight[n,:,:,:] = thisData - thisData * t / normData
            neurBias[n] = thisBias - thisBias * t / normData
          else:
            neurWeight[n,:,:,:] = 0
            neurBias[n] = 0

      if cuda: 
        thisMod.weight.data = torch.from_numpy( neurWeight ).cuda()
        thisMod.bias.data = torch.from_numpy( neurBias ).cuda()
      else:
        thisMod.weight.data = torch.from_numpy( neurWeight )
        thisMod.bias.data = torch.from_numpy( neurBias )


def proxL2LHalf( net, t, cuda ):
  for thisMod in net.modules():
    if hasattr( thisMod, 'weight' ):
      neurWeight = thisMod.weight.data.numpy()
      neurBias = thisMod.bias.data.numpy()
      if isinstance( thisMod, torch.nn.modules.linear.Linear ):
        for n in range(0,len(neurBias)):
          thisData = neurWeight[n,:]
          thisBias = neurBias[n]
          normData = np.sqrt( np.sum(thisData*thisData) + np.sum(thisBias*thisBias) )

          if normData == 0:
            neurWeight[:] = 0
            neurBias[n] = 0
          else :
            alpha = t / np.power( normData, 1.5 )
            if alpha < 2*np.sqrt(6)/9:
              s = 2 / np.sqrt(3) * np.sin( 1/3 * np.arccos( 3 * np.sqrt(3)/4 * alpha ) + math.pi/2 )
              neurWeight[n,:] = (s*s) * thisData 
              neurBias[n] = (s*s) * thisBias 
            else:
              neurWeight[:] = 0
              neurBias[n] = 0

      elif isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        for n in range(0,len(neurBias)):
          thisData = neurWeight[n,:,:,:]
          thisBias = neurBias[n]
          normData = np.sqrt( np.sum(thisData*thisData) + np.sum(thisBias*thisBias) )

          if normData == 0:
            neurWeight[n,:,:,:] = 0
            neurBias[n] = 0
          else :
            alpha = t / np.power( normData, 1.5 )
            if alpha < 2*np.sqrt(6)/9:
              s = 2 / np.sqrt(3) * np.sin( 1/3 * np.arccos( 3 * np.sqrt(3)/4 * alpha ) + math.pi/2 )
              neurWeight[n,:,:,:] = (s*s) * thisData 
              neurBias[n] = (s*s) * thisBias 
            else:
              neurWeight[n,:,:,:] = 0
              neurBias[n] = 0

      if cuda:
        thisMod.weight.data = torch.from_numpy( neurWeight ).cuda()
        thisMod.bias.data = torch.from_numpy( neurBias ).cuda()
      else:
        thisMod.weight.data = torch.from_numpy( neurWeight )
        thisMod.bias.data = torch.from_numpy( neurBias )


def saveCheckpoint( net, filename ):
  torch.save( net.state_dict(), filename )


def showResults( net, dataLoader, cuda ):
  # Determine accuracy on test set

  accuracy = findAccuracy( net, dataLoader, cuda )
  print( 'Accuracy of the network on the 10000 test inputs: %d %%' % ( 100 * accuracy ) )

  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  for data in dataLoader:
    inputs, labels = data
    if cuda:
      inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i]
      class_total[label] += 1

  for i in range(10):
    print('Accuracy of %5s : %2d %%' % ( classes[i], 100 * class_correct[i] / class_total[i]))


def softThreshTwoWeights( net, tConv, tLinear ):
  # Apply a soft threshold to the weights of a nn.Module object
  # Applies one weight to the convolutional layers and another to the fully connected layers
  for thisMod in net.modules():
    if hasattr( thisMod, 'weight' ):
      neurWeight = thisMod.weight.data.cpu().numpy()
      neurBias = thisMod.bias.data.cpu().numpy()
      if isinstance( thisMod, torch.nn.modules.conv.Conv2d ):
        neurWeight = np.sign( neurWeight ) * np.clip( np.abs(neurWeight) - tConv, 0, None )
        neurBias = np.sign( neurBias ) * np.clip( np.abs(neurBias) - tConv, 0, None )
      elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
        neurWeight = np.sign( neurWeight ) * np.clip( np.abs(neurWeight) - tLinear, 0, None )
        neurBias = np.sign( neurBias ) * np.clip( np.abs(neurBias) - tLinear, 0, None )

      if cuda: 
        thisMod.weight.data = torch.from_numpy( neurWeight ).cuda()
        thisMod.bias.data = torch.from_numpy( neurBias ).cuda()
      else:
        thisMod.weight.data = torch.from_numpy( neurWeight )
        thisMod.bias.data = torch.from_numpy( neurBias )


def softThreshWeights( net, t, cuda ):
  for thisMod in net.modules():
    if hasattr( thisMod, 'weight' ):
      neurWeight = thisMod.weight.data.cpu().numpy()
      neurBias = thisMod.bias.data.cpu().numpy()
      neurWeight = np.sign( neurWeight ) * np.clip( np.abs(neurWeight) - t, 0, None )
      neurBias = np.sign( neurBias ) * np.clip( np.abs(neurBias) - t, 0, None )
      if cuda: 
        thisMod.weight.data = torch.from_numpy( neurWeight ).cuda()
        thisMod.bias.data = torch.from_numpy( neurBias ).cuda()
      else:
        thisMod.weight.data = torch.from_numpy( neurWeight )
        thisMod.bias.data = torch.from_numpy( neurBias )


def trainWithAdam( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches

  optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch+1) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      if params.cuda:
        inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0] * len(dataLoader)

      optimizer.step()

      if k % params.showAccuracyEvery == params.showAccuracyEvery-1:
        testAccuracy = findAccuracy( net, testLoader, params.cuda )
        print( '[%d,%d] cost: %.3f,  testAccuracy: %.3f%%' % \
          ( epoch+1, i+1, costs[k], testAccuracy*100 ) )
      elif k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f' % ( epoch+1, i+1, costs[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return costs


def trainWithProxGradDescent_regL1Norm( dataLoader, net, criterion, params, learningRate ):
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

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    optimizer.zero_grad()
    for i, data in enumerate( dataLoader, 0 ):
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
    softThreshWeights( net, learningRate*regParam/nWeights )

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
      print( '[%d] cost: %.3f, sparsity: %d' % (epoch+1, costs[k], sparses[epoch] ) )

    k += 1

  return ( costs, sparses )


def trainWithProxGradDescent_regL2L1Norm( dataLoader, net, criterion, params, learningRate ):
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

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    optimizer.zero_grad()
    for i, data in enumerate( dataLoader, 0 ):
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
    proxL2L1( net, t=learningRate*regParam/nWeights, cuda=params.cuda )

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
      print( '[%d] cost: %.3f, groupSparsity: %d' % (epoch+1, costs[k], groupSparses[epoch] ) )

    k += 1

  return ( costs, groupSparses )


def trainWithStochProxGradDescent_regL1Norm( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL1

  nWeights = findNumWeights( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  sparses = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      optimizer.zero_grad()
      loss.backward()

      # Perform a gradient descent update
      optimizer.step()

      # Perform a proximal operator update
      softThreshWeights( net, learningRate*regParam/nWeights, cuda=params.cuda )

      # Determine the current objective function's value
      mainLoss = 0
      outputs = net(inputs)
      thisLoss = criterion(outputs, labels)
      mainLoss += thisLoss.data[0]
      regLoss = 0
      for W in net.parameters():
        regLoss = regLoss + W.norm(1)
      regLoss = torch.mul( regLoss, regParam/nWeights )
      costs[k] = mainLoss + regLoss.data[0]
      sparses[k] = findNumZeroWeights( net )

      if k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f, sparsity: %d' % ( epoch+1, i+1, costs[k], sparses[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, sparses )


def trainWithStochProxGradDescent_regL2L1Norm( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL2L1

  nNeurons = findNumNeurons( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  groupSparses = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  regLosses = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      dataInputs, dataLabels = data
      if params.cuda:
        dataInputs, dataLabels = dataInputs.cuda(async=True), dataLabels.cuda(async=True)
      inputs, labels = Variable(dataInputs), Variable(dataLabels)

      # Estimate the gradient using a single minibatch
      outputs = net( inputs )
      loss = torch.mul( criterion( outputs, labels ), 1/len(inputs) )
      optimizer.zero_grad()
      loss.backward()

      # Determine the current objective function's value
      regLoss = 0
      for thisMod in net.modules():
        if hasattr( thisMod, 'weight' ):
          neurWeight = thisMod.weight.data.cpu().numpy()
          neurBias = thisMod.bias.data.cpu().numpy()
          regLoss += np.sqrt( np.sum( neurWeight * neurWeight ) + np.sum( neurBias * neurBias ) )
      regLoss *= regParam/nNeurons
      regLosses[k] = regLoss
      costs[k] = loss.data[0] + regLoss
      groupSparses[k] = findNumDeadNeurons( net )

      # Determine the current batch accuracy
      _, predicted = torch.max( outputs.data, 1 )  # max along dimension 1
      correct = ( predicted == dataLabels ).sum()
      batchAccuracy = correct / dataLabels.size(0)

      if k % params.showAccuracyEvery == params.showAccuracyEvery-1:
        testAccuracy = findAccuracy( net, testLoader, params.cuda )
        trainAccuracy = findAccuracy( net, trainLoader, params.cuda )
        print( '[%d,%d] nNeurons: %d,  cost: %.8f,  regLoss: %.8f,  groupSparses %d,  ----------------  trainAccuracy: %.3f%%,  testAccuracy: %.3f%%' % \
          ( epoch+1, i+1, nNeurons, costs[k], regLoss, groupSparses[k], trainAccuracy*100, testAccuracy*100 ) )
      elif k % params.printEvery == params.printEvery-1 or k == 0:
        print( '[%d,%d] nNeurons: %d,  cost: %.8f,  regLoss: %.8f,  groupSparses: %d,  batchAccuracy: %.3f%%' % \
            ( epoch+1, i+1, nNeurons, costs[k], regLoss, groupSparses[k], batchAccuracy*100 ) )
      k += 1

      # Perform a gradient descent update
      optimizer.step()

      # Perform a proximal operator update
      proxL2L1( net, t=learningRate*regParam/nNeurons, cuda=params.cuda )

      if i >= nBatches-1:
        break

  return ( costs, groupSparses, regLosses )


def trainWithStochProxGradDescent_regL2L1Norm_varyingStepSize( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL2L1

  nNeurons = findNumNeurons( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  groupSparses = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      if params.cuda:
        inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net( inputs )
      loss = torch.mul( criterion( outputs, labels ), 1/len(inputs) )
      optimizer.zero_grad()
      loss.backward()

      # Perform a gradient descent update
      for param_group in optimizer.param_groups:
        #thisLearningRate = np.random.rayleigh( learningRate )
        #thisLearningRate = np.random.exponential( learningRate )
        #thisLearningRate = np.random.gamma( 10, learningRate )
        #thisLearningRate = np.random.uniform( 0, learningRate )
        thisLearningRate = np.absolute( np.random.laplace( 0, learningRate ) )
        param_group['lr'] = thisLearningRate 
      optimizer.step()

      # Perform a proximal operator update
      proxL2L1( net, t=learningRate*regParam/nNeurons, cuda=params.cuda )

      # Determine the current objective function's value
      mainLoss = torch.mul( criterion( outputs, labels ), 1/len(inputs) )
      regLoss = 0
      for thisMod in net.modules():
        if hasattr( thisMod, 'weight' ):
          neurWeight = thisMod.weight.data.cpu().numpy()
          neurBias = thisMod.bias.data.cpu().numpy()
          regLoss += np.sqrt( np.sum( neurWeight * neurWeight ) + np.sum( neurBias * neurBias ) )
      regLoss *= regParam/nNeurons
      costs[k] = mainLoss.data[0] + regLoss
      groupSparses[k] = findNumDeadNeurons( net )

      if k % params.showAccuracyEvery == params.showAccuracyEvery-1:
        testAccuracy = findAccuracy( net, testLoader, params.cuda )
        trainAccuracy = findAccuracy( net, trainLoader, params.cuda )
        print( '[%d,%d] cost: %.6f,  regLoss: %.5f,  groupSparses %d,  learningRate: %.5f,  ' + \
          'trainAccuracy: %.3f%%,  testAccuracy: %.3f%%' % \
          ( epoch+1, i+1, costs[k], regLoss, groupSparses[k], learningRate, \
          trainAccuracy*100, testAccuracy*100 ) )
      elif k % params.printEvery == params.printEvery-1 or k == 0:
        print( '[%d,%d] cost: %.6f,  regLoss: %.5f,  groupSparses %d,  learningRate: %.5f' % \
            ( epoch+1, i+1, costs[k], regLoss, groupSparses[k], thisLearningRate ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithStochProxGradDescentLS_regL2L1Norm( dataLoader, net, criterion, params, learningRate=1 ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL2L1
  r = params.r   # step size shrining factor
  s = params.s   # step size growing factor
  minStepSize = 0.01

  nNeurons = findNumNeurons( net )
  #optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  lastT = learningRate
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  groupSparses = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  stepSizes = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      if params.cuda:
        inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net( inputs )
      loss = torch.mul( criterion( outputs, labels ), 1/len(inputs) )
      optimizer = optim.SGD( net.parameters(), lr=lastT )
      optimizer.zero_grad()
      loss.backward()

      preDict = {k:v.clone() for k,v in net.state_dict().items()}
      t = s * lastT
      shrinkIter = 0
      while True:
        net.load_state_dict( preDict )
        thisStepSize = max( t, minStepSize )

        # Perform a gradient descent update
        #for param_group in optimizer.param_groups:
        #  param_group['lr'] = thisStepSize
        optimizer = optim.SGD( net.parameters(), lr=thisStepSize )
        optimizer.step()

        # Perform a proximal operator update
        proxL2L1( net, t=max(t,minStepSize)*regParam/nNeurons, cuda=params.cuda )

        postOutputs = net( inputs )
        postLoss = torch.mul( criterion( postOutputs, labels ), 1/len(inputs) )

        normL2Loss = 0
        dpLoss = 0
        for paramName, postParamValue in net.named_parameters():
          preGradArray = postParamValue.grad.data.cpu().numpy()  # note that the gradient hasn't been updated,
                                                           # so this is preGradArray.
          preValueArray = preDict[ paramName ].cpu().numpy()
          postValueArray = postParamValue.data.cpu().numpy()
          paramDiff = postValueArray - preValueArray
          dpLoss += np.sum( preGradArray * paramDiff )
          #normL2Loss += np.sum( np.square( paramDiff ) )
        #normL2Loss = normL2Loss / (2*t)

        if postLoss.data[0] <= loss.data[0] + dpLoss + normL2Loss or t < minStepSize:
          lastT = thisStepSize
          stepSizes[k] = thisStepSize
          break
        t *= r
        shrinkIter += 1

        #if (shrinkIter-1) % 10 == 9:
        #  print( "    Shrink Iter: " + str(shrinkIter) + "  Step size is: " + str(t) )


      # Determine the current objective function's value
      regLoss = 0
      for thisMod in net.modules():
        if hasattr( thisMod, 'weight' ):
          neurWeight = thisMod.weight.data.cpu().numpy()
          neurBias = thisMod.bias.data.cpu().numpy()
          regLoss += np.sqrt( np.sum( neurWeight * neurWeight ) + np.sum( neurBias * neurBias ) )
      regLoss *= regParam/nNeurons
      costs[k] = postLoss.data[0] + regLoss
      groupSparses[k] = findNumDeadNeurons( net )

      if k % params.showAccuracyEvery == params.showAccuracyEvery-1:
        testAccuracy = findAccuracy( net, testLoader, params.cuda )
        trainAccuracy = findAccuracy( net, trainLoader, params.cuda )
        print( '[%d,%d] preLoss: %.10f,  postLoss: %.10f,  cost: %.6f,  regLoss: %.5f,  groupSparses %d,  stepSize: %.8f,  trainAccuracy: %.3f%%,  testAccuracy: %.3f%%' % \
          ( epoch+1, i+1, loss.data[0], postLoss.data[0], costs[k], regLoss, groupSparses[k], thisStepSize, trainAccuracy*100, testAccuracy*100 ) )
      elif k % params.printEvery == params.printEvery-1 or k == 0:
        print( '[%d,%d] preLoss: %.10f,  postLoss: %.10f,  cost: %.6f,  regLoss: %.5f,  groupSparses %d,  stepSize: %.8f' % \
          ( epoch+1, i+1, loss.data[0], postLoss.data[0], costs[k], regLoss, groupSparses[k], thisStepSize ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses, stepSizes )


def trainWithStochProxGradDescent_regL2LHalfNorm( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  nBatches = params.nBatches
  regParam = params.regParam_normL2L1

  nWeights = findNumWeights( net )
  optimizer = optim.SGD( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  groupSparses = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      if params.cuda:
        inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net( inputs )
      loss = criterion( outputs, labels )
      optimizer.zero_grad()
      loss.backward()

      # Perform a gradient descent update
      optimizer.step()

      # Perform a proximal operator update
      net.apply( lambda w: proxL2LHalf( w, t=learningRate*regParam/nWeights, cuda=params.cuda ) )

      # Determine the current objective function's value
      mainLoss = criterion( outputs, labels )
      regLoss = 0
      for thisMod in net.modules():
        if hasattr( thisMod, 'weight' ):
          neurWeight = thisMod.weight.data.cpu().numpy()
          neurBias = thisMod.bias.data.cpu().numpy()
          regLoss = regLoss + np.sqrt( np.sqrt( \
            np.sum( neurWeight * neurWeight  ) + np.sum( neurBias * neurBias ) ) )
      regLoss = regLoss * regParam/nWeights
      loss = mainLoss + regLoss
      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )

      if k % params.showAccuracyEvery == params.showAccuracyEvery-1:
        testAccuracy = findAccuracy( net, testLoader, params.cuda )
        trainAccuracy = findAccuracy( net, trainLoader, params.cuda )
        print( '[%d,%d] cost: %.3f,  regLoss: %.3f,  trainAccuracy: %.3f%%,  testAccuracy: %.3f%%' % \
          ( epoch+1, i+1, costs[k], regLoss, trainAccuracy*100, testAccuracy*100 ) )
      elif k % params.printEvery == params.printEvery-1:
        print( '[%d,%d] cost: %.3f,  regLoss: %.3f' % ( epoch+1, i+1, costs[k], regLoss ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithStochSubGradDescent( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  nNeurons = findNumNeurons( net )
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      dataInputs, dataLabels = data
      if params.cuda:
        dataInputs, dataLabels = dataInputs.cuda(async=True), dataLabels.cuda(async=True)
      inputs, labels = Variable(dataInputs), Variable(dataLabels)

      # Estimate the gradient using a single minibatch
      outputs = net(inputs)
      loss = torch.mul( criterion(outputs, labels), 1/params.batchSize )
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0]

      optimizer.step()

      # Determine the current batch accuracy
      _, predicted = torch.max( outputs.data, 1 )  # max along dimension 1
      correct = ( predicted == dataLabels ).sum()
      batchAccuracy = correct / dataLabels.size(0)

      if k % params.showAccuracyEvery == params.showAccuracyEvery-1:
        testAccuracy = findAccuracy( net, testLoader, params.cuda )
        trainAccuracy = findAccuracy( net, trainLoader, params.cuda )
        print( '[%d,%d] nNeurons: %d,  cost: %.6f,  trainAccuracy: %.3f%%,  testAccuracy: %.3f%%' % \
          ( epoch+1, i+1, nNeurons, costs[k], trainAccuracy*100, testAccuracy*100 ) )
      elif k % params.printEvery == params.printEvery-1 or k == 0:
        print( '[%d,%d] nNeurons: %d,  cost: %.6f,  batchAccuracy: %.3f%%' % ( epoch+1, i+1, nNeurons, costs[k], batchAccuracy*100 ) )
      k += 1

      if i >= nBatches-1:
        break

  return costs


def trainWithStochSubGradDescent_regL1Norm( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL1

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  nWeights = findNumWeights(net)

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  sparses = [None] * (nEpochs * np.min([len(dataLoader), nBatches]))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
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
        print( '[%d,%d] cost: %.3f, sparsity: %d' % ( epoch+1, k+1, costs[k], sparses[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return (costs,sparses)


def trainWithStochSubGradDescent_regL2L1Norm( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL2L1

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  nWeights = findNumWeights(net)

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  groupSparses = [None] * (nEpochs * np.min([len(dataLoader), nBatches]))
  groupAlmostSparses = [None] * (nEpochs * np.min([len(dataLoader), nBatches]))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      mainLoss = torch.mul( criterion( outputs, labels ), 1/params.batchSize )
      regLoss = Variable( torch.FloatTensor(1), requires_grad=True )
      for thisMod in net.modules():

        if isinstance(thisMod, torch.nn.modules.conv.Conv2d):
          neurWeight = thisMod.weight
          neurBias = thisMod.bias
          nNeurons = neurWeight.shape[0]
          for n in range(0,nNeurons):
            regLoss = regLoss + torch.sqrt( torch.mul( neurWeight[n,:,:,:].norm(2), neurWeight[n,:,:,:].norm(2) ) + \
              torch.mul( neurBias[n], neurBias[n] ) )

        elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
          neurWeight = thisMod.weight
          neurBias = thisMod.bias
          nNeurons = neurWeight.shape[0]
          for n in range(0,nNeurons):
            regLoss = regLoss + torch.sqrt( torch.mul( neurWeight[n,:].norm(2), neurWeight[n,:].norm(2) ) + \
              torch.mul( neurBias[n], neurBias[n] ) )

      loss = mainLoss + torch.mul( regLoss, regParam/nWeights )
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )
      groupAlmostSparses[k] = findNumDeadNeurons( net, thresh=1e-6 )

      optimizer.step()

      if k % printEvery == printEvery-1:
        print( '[%d,%d] cost: %.3f,  group sparsity: %d,  group almost sparsity: %d' % \
          ( epoch+1, i+1, costs[k], groupSparses[k], groupAlmostSparses[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses, groupAlmostSparses )


def trainWithStochSubGradDescent_regL2LHalfNorm( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery
  regParam = params.regParam_normL2L1

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  nWeights = findNumWeights(net)

  k = 0
  costs = [None] * ( nEpochs * np.min([len(dataLoader),nBatches]) )
  groupSparses = [None] * (nEpochs * np.min([len(dataLoader), nBatches]))
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    for i, data in enumerate( dataLoader, 0 ):
      inputs, labels = data
      if params.cuda:
        inputs, labels = inputs.cuda(async=True), labels.cuda(async=True)
      inputs, labels = Variable(inputs), Variable(labels)

      # Calculate the gradient using just a minibatch
      outputs = net(inputs)
      mainLoss = criterion( outputs, labels )
      regLoss = Variable( torch.FloatTensor(1), requires_grad=True )
      for thisMod in net.modules():

        if isinstance(thisMod, torch.nn.modules.conv.Conv2d):
          neurWeight = thisMod.weight
          neurBias = thisMod.bias
          nNeurons = neurWeight.shape[0]
          for n in range(0,nNeurons):
            regLoss = regLoss + torch.sqrt( torch.sqrt( \
              torch.sum( torch.mul( neurWeight[n,:,:,:].norm(2), neurWeight[n,:,:,:].norm(2) ) ) + \
              torch.sum( torch.mul( neurBias[n], neurBias[n] ) ) ) )

        elif isinstance( thisMod, torch.nn.modules.linear.Linear ):
          neurWeight = thisMod.weight
          neurBias = thisMod.bias
          nNeurons = neurWeight.shape[0]
          for n in range(0,nNeurons):
            regLoss = regLoss + torch.sqrt( torch.sqrt( \
              torch.sum( torch.mul( neurWeight[n,:].norm(2), neurWeight[n,:].norm(2) ) ) + \
              torch.sum( torch.mul( neurBias[n], neurBias[n] ) ) ) )

      loss = mainLoss + torch.mul( regLoss, regParam/nWeights )
      optimizer.zero_grad()
      loss.backward()

      costs[k] = loss.data[0]
      groupSparses[k] = findNumDeadNeurons( net )

      optimizer.step()

      if k % printEvery == printEvery-1:
        print( '[%d,%d] cost: %.3f,  group sparsity: %d' % \
          ( epoch+1, i+1, costs[k], groupSparses[k] ) )
      k += 1

      if i >= nBatches-1:
        break

  return ( costs, groupSparses )


def trainWithSubGradDescent( dataLoader, net, criterion, params, learningRate ):
  nEpochs = params.nEpochs
  momentum = params.momentum
  nBatches = params.nBatches
  printEvery = params.printEvery

  optimizer = optim.SGD( net.parameters(), lr=learningRate, momentum=momentum )
  #optimizer = optim.Adam( net.parameters(), lr=learningRate )

  k = 0
  costs = [None] * nEpochs
  for epoch in range(nEpochs):  # loop over the dataset multiple times

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    optimizer.zero_grad()
    loss = 0
    for i, data in enumerate( dataLoader, 0 ):
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

    if epoch % params.saveCheckpointEvery == params.saveCheckpointEvery-1:
      saveCheckpoint( net, params.checkpointDir + '/checkpoint_' + str(epoch) + '.net' )

    optimizer = optim.SGD( net.parameters(), lr=lastT )

    # Evaluate the loss function before the gradient descent step
    preLoss = 0
    optimizer.zero_grad()
    for i, data in enumerate( dataLoader, 0 ):
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
        #multi_setattr( net, paramName+'.grad', thisGrad/len(dataLoader) )

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
      for i, data in enumerate( dataLoader, 0 ):
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
class smartNet( nn.Module ):
  layerNames = [ "conv1", "conv2", "conv3", "fc1", "fc2", "fc3" ]

  def __init__(self,layerSizes):
    # layerSizes is a list of integers specifying the sizes of the first 5 layers
    super(smartNet, self).__init__()
    self.conv1 = nn.Conv2d( 3, layerSizes[0], 5 )   # inChannels, outChannels, kSize
    self.conv2 = nn.Conv2d( layerSizes[0], layerSizes[1], 3 )
    self.conv3 = nn.Conv2d( layerSizes[1], layerSizes[2], 3 )
    self.fc1 = nn.Linear( layerSizes[2] * 2 * 2, layerSizes[3] )   # inChannels, outChannels
    self.fc2 = nn.Linear( layerSizes[3], layerSizes[4] )
    self.fc3 = nn.Linear( layerSizes[4], 10 )

  def forward(self, x):
    x = F.avg_pool2d( F.softplus( self.conv1(x), beta=100 ), 2, 2 )
    x = F.avg_pool2d( F.softplus( self.conv2(x), beta=100 ), 2, 2 )
    x = F.avg_pool2d( F.softplus( self.conv3(x), beta=100 ), 2, 2 )
    xShape = x.data.cpu().numpy().shape
    x = x.view( -1, int( np.prod( xShape[1:4] ) ) )  # converts matrix to vector
    x = F.softplus( self.fc1(x), beta=100 )
    x = F.softplus( self.fc2(x), beta=100 )
    x = self.fc3( x )
    x = F.log_softmax( x, dim=1 )
    return x

class Net(nn.Module):
  layerNames = [ "conv1", "conv2", "conv3", "fc1", "fc2", "fc3" ]

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d( 3, 300, 5 )   # inChannels, outChannels, kSize
    self.conv2 = nn.Conv2d( 300, 200, 3 )
    self.conv3 = nn.Conv2d( 200, 100, 3 )
    self.fc1 = nn.Linear( 100 * 2 * 2, 500 )   # inChannels, outChannels
    self.fc2 = nn.Linear( 500, 200 )
    self.fc3 = nn.Linear( 200, 10 )

  def forward(self, x):
    x = F.avg_pool2d( F.softplus( self.conv1(x), beta=100 ), 2, 2 )
    x = F.avg_pool2d( F.softplus( self.conv2(x), beta=100 ), 2, 2 )
    x = F.avg_pool2d( F.softplus( self.conv3(x), beta=100 ), 2, 2 )
    x = x.view( -1, 100 * 2 * 2 )  # converts matrix to vector
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
# What are trainLoader, testLoader?  How can I use other datasets?
  # Look at "Classifying Images of Hand Signs" to make dataset and dataLoader objects
# Why doesn't the example show the CIFAR10 images when running using PyCharm (only shows in debug mode)?
# Why doesn't the example show images when running from the command line?



# Parameters for this code
class Params:
  batchSize = 1000
  checkpointDir = 'checkpoints'
  cuda = 0
  datacase = 0
  learningRate = 0.05
  momentum = 0.0
  nBatches = 1000000
  nEpochs = 500
  printEvery = 5
  regParam_normL1 = 0e1
  regParam_normL2L1 = 0e1
  regParam_normL2Lhalf = 0e1
  saveCheckpointEvery = 100  # save state every this many epochs
  seed = 1
  showAccuracyEvery = 200
  shuffle = False  # Shuffle the data in each minibatch
  alpha = 0.8
  r = 0.9  # Backtracking line search parameter (must be between 0 and 1)
  s = 1.5  # Step size scaling parameter (must be greater than 1)
  #warmStartFile = None
  warmStartFile = './results/ssgResults.net'
  #warmStartFile = './results/results0.net'


if __name__ == '__main__':

  params = Params()
  torch.manual_seed( params.seed )

  params.cuda = torch.cuda.is_available()
  if params.cuda: torch.cuda.manual_seed( params.seed )

  ( trainset, trainLoader, testset, testLoader, classes ) = loadData( \
    params.datacase, params.batchSize, params.shuffle )

  #net = Net()
  layerSizes = [ 500, 500, 500, 400, 300 ]
  net = smartNet( layerSizes )
  net = net.cuda() if params.cuda else net
  if params.warmStartFile is not None:
    loadCheckpoint( net, params.warmStartFile )

  #net = pruneNet( net, params.cuda )

  print( "Num Neurons: %d" % findNumNeurons( net ) )

  if not os.path.isdir( params.checkpointDir ):
    os.mkdir( params.checkpointDir )
  if not os.path.isdir( 'results' ):
    os.mkdir( 'results' )

  # get some random training images
  #dataiter = iter( trainLoader )
  #images, labels = dataiter.next()
  #imshow( torchvision.utils.make_grid(images) )  # show images
  #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))  # print labels

  # Tests
  #nLive = findNumLiveNeuronsInLayers( net )  # test of findNumLiveNeuronsInLayers
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
  #costs = trainWithSubGradDescent( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #costs = trainWithAdam( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #costs = trainWithSubGradDescentLS( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #costs = trainWithStochSubGradDescent( trainLoader, net, criterion, params, learningRate=params.learningRate )  # works with batch size of 1000 and step size of 0.1

  # L1 norm regularization
  #(costs,sparses) = trainWithStochSubGradDescent_regL1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs,sparses) = trainWithStochProxGradDescent_regL1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs, sparses) = trainWithProxGradDescent_regL1Norm(trainLoader, net, criterion, params, learningRate=params.learningRate )

  # L2,L1 norm regularization
  #(costs,groupSparses) = trainWithProxGradDescent_regL2L1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs,groupSparses,groupAlmostSparses) = trainWithStochSubGradDescent_regL2L1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs,groupSparses) = trainWithStochProxGradDescent_regL2L1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs,groupSparses) = trainWithStochProxGradDescent_regL2L1Norm_varyingStepSize( \
  #  trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs,groupSparses,stepSizes) = trainWithStochProxGradDescentLS_regL2L1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )

  #L2,L1/2 norm regularization
  #(costs,groupSparses) = trainWithStochSubGradDescent_regL2LHalfNorm( trainLoader, net, criterion, params, learningRate=params.learningRate )
  #(costs,groupSparses) = trainWithStochProxGradDescent_regL2LHalfNorm( trainLoader, net, criterion, params, learningRate=params.learningRate )


  baseCheckpointDir = params.checkpointDir
  #costs = trainWithStochSubGradDescent( trainLoader, net, criterion, params, learningRate=params.learningRate )  # works with batch size of 1000 and step size of 0.1
  #torch.save( net.state_dict(), './results/ssgResults.net' )
  for regPower in range(4,20):
    params.checkpointDir = baseCheckpointDir + str(regPower)
    if not os.path.isdir( params.checkpointDir ):
      os.mkdir( params.checkpointDir )

    params.regParam_normL2L1 = 2 ** regPower
    #params.learningRate = 5 * 10 ** -(regPower+2)

    print( "Working on regPower: " + str(regPower) + ",  regParam: " + str(params.regParam_normL2L1) + ",  learningRate: " + str(params.learningRate) )

    (costs,groupSparses,regLosses) = trainWithStochProxGradDescent_regL2L1Norm( trainLoader, net, criterion, params, learningRate=params.learningRate )
    torch.save( net.state_dict(), './results/trained_' + str(params.regParam_normL2L1 + '.net') )

    pruneNet( net, params.cuda )
    nNeurons = findNumNeurons( net )
    print( "After pruning, the number of neurons is: " + str(nNeurons) )
    costs = trainWithStochSubGradDescent( trainLoader, net, criterion, params, learningRate=params.learningRate )  # works with batch size of 1000 and step size of 0.1

    trainAccuracy = findAccuracy( net, trainLoader, params.cuda )
    testAccuracy = findAccuracy( net, testLoader, params.cuda )

    with open( './results/results' + str(regPower) + '.pkl', 'wb') as f:
      pickle.dump( [ trainAccuracy, testAccuracy, costs, groupSparses, regLosses ], f )
    torch.save( net.state_dict(), './results/results' + str(regPower) + '.net' )



  # Prune the network and re-train
  nLives = findNumLiveNeuronsInLayers( net )  # test of findNumLiveNeuronsInLayers




  #with open( 'trainWithStochProxGradDescent_regL2L1Norm_1pt0.pkl', 'rb' ) as f:
  #  testAccuracy, costs, groupSparses = pickle.load( f )
  #net.load_state_dict( torch.load( 'trainWithStochProxGradDescent_regL2L1Norm_1pt0.net' ) )



  #plt.plot( costs )
  #plt.title('Stochastic Proximal Gradient with L2,L1 Regularization 1.0')
  #plt.show()

  #plt.plot( groupSparses )
  #plt.title('Stochastic Proximal Gradient with L2,L1 Regularization 1.0')
  #plt.show()



  print("Test Results:")
  showResults( net, testLoader, params.cuda )

  print("Train Results:")
  showResults( net, trainLoader, params.cuda )

  dataiter = iter(testLoader)
  images, labels = dataiter.next()

  # print images
  imshow( torchvision.utils.make_grid(images) )
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


  outputs = net( Variable(images) )
  _, predicted = torch.max( outputs.data, 1 )

  print( 'Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)) )


