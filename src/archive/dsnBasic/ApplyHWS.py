import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pdb
import skimage.io as skio
from scipy.signal import medfilt as med_filt
import math
import random
import skimage.transform
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from scipy.ndimage.measurements import label
import udsn
from unet import UNet
from extra import ScaleAndShow
from GdalLoader import LoadSyn
from SynData import GenSyn1


##############################################################
## Basic Training Program
if __name__ == '__main__':
    verbose = 0
    showPics = True
    theSeed = 0
    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    numTrain = 10
    lossType = 'purity'
    #X = LoadSyn()
    X = GenSyn1()
    W = X.shape[0]
    H = X.shape[1]

    ScaleAndShow(X, 1)
    plt.show()
        
    wimg, cimg, G = udsn.ApplyDSN(X)
    imgCC = udsn.GetLabeledImage(G, -1.0, W, H, 'weight')
    ScaleAndShow(imgCC, 4)
    
    #imgX, imgY = udsn.GetGraphImages(G, W, H, 'minmax')
    G = udsn.MinMaxWatershedCut(G)
    imgWS = udsn.GetLabeledImage(G, -1.0, W, H, 'minmax')
    #ScaleAndShow(imgX, 2)
    #ScaleAndShow(imgY, 3)
    
    ScaleAndShow(imgWS, 5)

    #ScaleAndShow(wimg, 4)
    #ScaleAndShow(cimg, 5)
    plt.show()



