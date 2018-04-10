
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

from nnOpt import *


### Function definitions ###

def imshow(img):  # function to show an image
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow( np.transpose( npimg, (1, 2, 0) ) )


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

def pruneSimpleNet( net, cuda, thresh=0 ):
  nLives = findNumLiveNeuronsInLayers( net, thresh=thresh )
  newNet = smartSimpleNet( nLives )

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



### Object definitions ###
class smartSimpleNet( nn.Module ):
  layerNames = [ "conv1", "conv2", "conv3", "fc1", "fc2", "fc3" ]

  def __init__(self,layerSizes):
    # layerSizes is a list of integers specifying the sizes of the first 5 layers
    super(smartSimpleNet, self).__init__()
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

class simpleNet(nn.Module):
  layerNames = [ "conv1", "conv2", "conv3", "fc1", "fc2", "fc3" ]

  def __init__(self):
    super(simpleNet, self).__init__()
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
  warmStartFile = None
  #warmStartFile = './results/ssgResults.net'
  #warmStartFile = './results/results0.net'


if __name__ == '__main__':

  params = Params()
  torch.manual_seed( params.seed )

  params.cuda = torch.cuda.is_available()
  if params.cuda: torch.cuda.manual_seed( params.seed )

  ( trainset, trainLoader, testset, testLoader, classes ) = loadData( \
    params.datacase, params.batchSize, params.shuffle )

  #net = simpleNet()
  layerSizes = [ 500, 500, 500, 400, 300 ]
  net = smartSimpleNet( layerSizes )
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
  costs = trainWithStochSubGradDescent( trainLoader, net, criterion, params, learningRate=params.learningRate )  # works with batch size of 1000 and step size of 0.1
  #torch.save( net.state_dict(), './results/ssgResults.net' )



  # Polishing: prune the network and re-train
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


