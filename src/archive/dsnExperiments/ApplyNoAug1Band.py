import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
from unet import UNet
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


def GetSegImages(WG, CG, W, H):
    imgWS = np.zeros((W, H), np.single)
    imgCC = np.zeros((W, H), np.single)
    
    for u, v, d in WG.edges(data = True):
        ul = WG.nodes[u]['label']
        imgWS[u[0], u[1]] = ul
        imgCC[u[0], u[1]] = CG.nodes[ul]['label']

        vl = WG.nodes[v]['label']
        imgWS[v[0], v[1]] = vl
        imgCC[v[0], v[1]] = CG.nodes[vl]['label']

    return(imgWS, imgCC)

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')


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

##############################################################
## Basic Training Program
if __name__ == '__main__':
    verbose = 0
    theSeed = 0

    numEpochs = 10000
    numTrain = 10
    numValid = 10  

    unetFeatures = 8
    unetDepth = 5

    learningRate = 1
    rhoMemory = 0.9
    rateStep = 100
    learningRateGamma = 0.7
    lossType = 'purity'
    saveNum = lossType
    useBatchNorm = True    

    np.random.seed(theSeed)
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    extractDir = "d:\\image_data\\snemi3d-extracts\\"
    imgType = "128_"
    limgType = "128"
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

    
    with open(trainName1, 'rb') as f:
        X = pickle.load(f)
    with open(trainNameLabel, 'rb') as f:
        Y = pickle.load(f)
    with open(validName1, 'rb') as f:
        XV = pickle.load(f)
    with open(validNameLabel, 'rb') as f:
        YV = pickle.load(f)

    X = X.astype(np.single)
    XV = XV.astype(np.single)
    Y = Y.astype(np.single)
    YV = YV.astype(np.single)

    #Ysyn = Ysyn.astype(np.single)

    XT = torch.tensor(X, requires_grad=False)                
    YT = torch.tensor(Y, requires_grad=False)                                                
    XVT = torch.tensor(XV, requires_grad=False)                
    YVT = torch.tensor(YV, requires_grad=False)                                                

    XT = torch.cat((XT, XVT), 0)
    YT = torch.cat((YT, YVT), 0)

    totalTrain = XT.shape[0]
    totalValid = XVT.shape[0]

    if numTrain < 0:
        numTrain = totalTrain
    if numValid < 0:
        numValid = totalValid

    print("Batch: " + str(numTrain) + " out of " + str(totalTrain))

    # Setting up U-net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNet(in_channels=1, n_classes=1, depth=unetDepth, wf=unetFeatures, padding=True, batch_norm=useBatchNorm, up_mode='upsample').to(device)
    model.load_state_dict(torch.load("train1D2_model_6000.pth"))
    #model = UNet(in_channels=1, n_classes=1, depth=unetDepth, wf=unetFeatures, padding=True, batch_norm=useBatchNorm, up_mode='upsample').to(device)
    #model.load_state_dict(torch.load("train1D2_model_6000.pth"))

    # Initializing training and validation lists to be empty
    val_loss_lst   = []
    val_rerror_lst  = []

    # Initializing best loss and random error
    val_best_loss   = math.inf
    val_best_rerror = math.inf

    # Validation every epoch        
    loss_lst_epoch   = []
    rerror_lst_epoch = []
    # Bar
    
    #model.eval()
    with torch.no_grad():                

        tia = np.random.permutation(totalTrain)
        #vi = np.random.permutation(totalValid)

        ti = tia[700:(700+numTrain)]
        np.random.shuffle(ti)
        
        for trn_idx in range(0, numTrain):
            print("Image "  + str(ti[trn_idx]) + "\n")
            
            X1 = XT[ ti[trn_idx] ]
            Y1 = YT[ ti[trn_idx] ]


            if True:
                X11 = X1.detach().numpy()    
                Y11 = Y1.detach().numpy()    
                ScaleAndShow(X11[0].squeeze(), 0)
                ScaleAndShow(Y11.squeeze(), 1)
                #plt.show()

            X1 = X1.unsqueeze(0)                
            X1 = X1.to(device)                         
            Y1 = Y1.squeeze()

            uOut = model(X1)
            if True:
                netOut = torch.squeeze(uOut).cpu()
                imgOut = netOut.detach().numpy()
                ScaleAndShow(imgOut, 2)
                #plt.show()
            
            loss, randError = dsn.RandLossDSN(uOut, Y1, lossType, trn_idx)    
            rerror_lst_epoch  = rerror_lst_epoch + [randError]
            loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]
            trn_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            trn_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)

            print("Loss   : " + str(loss.detach().numpy()) + "   Error: " + str(randError))
            print("TEST LOSS: " + str(trn_cepoch_loss) + " Error: " + str(trn_cepoch_rerror))

            wimg, cimg = dsn.ApplyDSN(uOut)
            ScaleAndShow(wimg, 4)
            ScaleAndShow(cimg, 5)
            fname = "seg_result_" + str(ti[trn_idx]) + ".pkl"
            with open(fname, 'wb') as f:
                pickle.dump(cimg, f)            
            fname = "unet_result_" + str(ti[trn_idx]) + ".pkl"
            with open(fname, 'wb') as f:
                pickle.dump(imgOut, f)            
            fname = "seg_gt_" + str(ti[trn_idx]) + ".pkl"
            with open(fname, 'wb') as f:
                pickle.dump(Y11.squeeze(), f)            

            plt.show()
            print("Continuing")
    
    

            #gtImg = Y1.detach().numpy()            
            #error = ExhaustiveRand(cimg, gtImg)
            

            #print("\t\t" + str(trn_idx) + " Loss " + str(loss.item()) + "  and Rand " + str(randError))                                                        
            #ScaleAndShow(X11[2].squeeze(), 0)
            #ScaleAndShow(X11[0].squeeze(), 1)
            #ScaleAndShow(gtImg, 2)
            #ScaleAndShow(imgOut, 3)

     
    # Loading model
    # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
    # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
    # uOut1 = model(X3)
    # uOut2 = loaded_model(X3)
