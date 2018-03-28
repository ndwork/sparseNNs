
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re   # Regular expressions
import torch

import sparseNNs

if __name__ == '__main__':

  net = sparseNNs.Net()
  netFile = './trainWithStochProxGradDescent_regL2L1Norm_0pt1.net'
  net.load_state_dict( torch.load( 'trainWithStochProxGradDescent_regL2L1Norm_0pt1.net' ) )


  params = sparseNNs.Params()
  resultFile = './trainWithStochProxGradDescent_regL2L1Norm_0pt1.pkl'
  with open( resultFile, 'rb' ) as f:
    testAccuracy, costs, groupSparses = pickle.load(f)

