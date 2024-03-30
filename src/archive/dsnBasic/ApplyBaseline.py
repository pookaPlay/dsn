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
import dsn
from unet import UNet
from extra import ScaleAndShow
from DataLoader import LoadPretrained

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

    [XT, YT, XVT, YVT] = LoadPretrained()
    
    print("Batch: " + str(numTrain) + " out of " + str(XT.shape[0]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    loss_lst_epoch   = []
    error_lst_epoch = []
    
    with torch.no_grad():                
        
        for trn_idx in range(3, 4):
            
            X1 = XT[ trn_idx ]
            Y1 = YT[ trn_idx ]

            X1 = X1.unsqueeze(0)                
            X1 = X1.to(device)                     
            Y1 = Y1.squeeze()

            uOut = X1
            
            if showPics:
                netOut = torch.squeeze(uOut).cpu()
                imgOut = netOut.detach().numpy()
                ScaleAndShow(imgOut, 1)
                #plt.show()
                gtImg = Y1.detach().numpy()            
                ScaleAndShow(gtImg, 2)                
            
            loss, randError = dsn.RandLossDSN(uOut, Y1, lossType, trn_idx)    
            print("Loss   : " + str(loss.detach().numpy()) + "   Error: " + str(randError))            

            error_lst_epoch  = error_lst_epoch + [randError]
            loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]
            trn_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            trn_error  = sum(error_lst_epoch)/len(error_lst_epoch)            
            print("==> AVG Loss: " + str(trn_loss) + " Error: " + str(trn_error))

            if showPics:
                wimg, cimg = dsn.ApplyDSN(uOut)
                ScaleAndShow(wimg, 4)
                ScaleAndShow(cimg, 5)
                plt.show()
                print("Continuing")

