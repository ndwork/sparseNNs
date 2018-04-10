
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

import myResnet


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


def loadCheckpoint( net, checkpointFile, costs=None ):
  net.load_state_dict( torch.load( checkpointFile ) )
  if costs is not None:
    with open( resultFile, 'rb' ) as f:
      testAccuracy, costs, groupSparses = pickle.load(f)


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



