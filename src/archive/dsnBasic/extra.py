import sys
import glob
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

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
        #simg = 1.0 - simg
    
    #plt.imshow(simg, cmap='gray')
    plt.imshow(simg, cmap='jet')



def ExhaustiveRand(seg, labels):
    W = seg.shape[0]
    H = seg.shape[1]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for j in range(H):
        for i in range(W):
            ind1 = j*W + i
            for jj in range(H):
                for ii in range(W):
                    ind2 = jj*W + ii
                    if ind1 < ind2:
                        if labels[i, j] == labels[ii, jj]:
                            TP = TP + 1
                            if seg[i, j] != seg[ii, jj]:
                                FN = FN + 1
                        else:
                            TN = TN + 1
                            if seg[i, j] == seg[ii, jj]:
                                FP = FP + 1
    error  = (FN + FP) / (TP + TN)
    print("Rand Error: " + str(error))
    print("From #pos: " + str(TP) + " #neg: " + str(TN))
    print("   and FN: " + str(FN) + "   FP: " + str(FP))
    return(error)

