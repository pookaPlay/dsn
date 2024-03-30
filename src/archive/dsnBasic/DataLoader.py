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

extractDir = "d:\\image_data\\snemi3d-extracts\\"
imgType = "128_"
limgType = "128"
theSeed = 0

trainName0 = extractDir + "train" + imgType + "0." + str(theSeed) + ".pkl"
trainName1 = extractDir + "train" + imgType + "1." + str(theSeed) + ".pkl"
trainName2 = extractDir + "train" + imgType + "2." + str(theSeed) + ".pkl"
trainName4 = extractDir + "train" + imgType + "4." + str(theSeed) + ".pkl"
trainNameLabel = extractDir + "trainLabel" + limgType + "." + str(theSeed) + ".pkl"

validName0 = extractDir + "valid" + imgType + "0." + str(theSeed) + ".pkl"
validName1 = extractDir + "valid" + imgType + "1." + str(theSeed) + ".pkl"
validName2 = extractDir + "valid" + imgType + "2." + str(theSeed) + ".pkl"
validName4 = extractDir + "valid" + imgType + "4." + str(theSeed) + ".pkl"
validNameLabel = extractDir + "validLabel" + limgType + "." + str(theSeed) + ".pkl"
    
def LoadPretrained():
    with open(trainName0, 'rb') as f:
        X = pickle.load(f)
    with open(trainNameLabel, 'rb') as f:
        Y = pickle.load(f)
    with open(validName0, 'rb') as f:
        XV = pickle.load(f)
    with open(validNameLabel, 'rb') as f:
        YV = pickle.load(f)

    X = X.astype(np.single)
    XV = XV.astype(np.single)
    Y = Y.astype(np.single)
    YV = YV.astype(np.single)


    XT = torch.tensor(X, requires_grad=False)                
    YT = torch.tensor(Y, requires_grad=False)                                                
    XVT = torch.tensor(XV, requires_grad=False)                
    YVT = torch.tensor(YV, requires_grad=False)                                                

    return(XT, YT, XVT, YVT)


def Load4Band():
    with open(trainName4, 'rb') as f:
        X = pickle.load(f)
    with open(trainNameLabel, 'rb') as f:
        Y = pickle.load(f)
    with open(validName4, 'rb') as f:
        XV = pickle.load(f)
    with open(validNameLabel, 'rb') as f:
        YV = pickle.load(f)

    X = X.astype(np.single)
    XV = XV.astype(np.single)
    Y = Y.astype(np.single)
    YV = YV.astype(np.single)


    XT = torch.tensor(X, requires_grad=False)                
    YT = torch.tensor(Y, requires_grad=False)                                                
    XVT = torch.tensor(XV, requires_grad=False)                
    YVT = torch.tensor(YV, requires_grad=False)                                                

    return(XT, YT, XVT, YVT)

