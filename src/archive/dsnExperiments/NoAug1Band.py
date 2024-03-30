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

# MATPLOTLIB defaults
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15


def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')

##############################################################
## Basic Training Program
if __name__ == '__main__':
    verbose = 0
    theSeed = 0

    numEpochs = 10000
    numTrain = 20
    numValid = 20  
    logScores = 100
    unetFeatures = 8
    unetDepth = 5
    lossType = 'purity'
    saveNum = lossType
    useBatchNorm = True

    learningRate = 1
    rhoMemory = 0.9
    rateStep = 100
    learningRateGamma = 0.7
    
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

    if numTrain < 0:
        numTrain = totalTrain

    print("Batch: " + str(numTrain) + " out of " + str(totalTrain))

    # Setting up U-net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=1, n_classes=1, depth=unetDepth, wf=unetFeatures, padding=True, batch_norm=useBatchNorm, up_mode='upsample').to(device)

    optimizer = optim.Adadelta(model.parameters(), rho=rhoMemory, lr=learningRate)
    
    # Initializing training and validation lists to be empty
    trn_loss_lst   = []
    trn_rerror_lst = []

    # Initializing best loss and random error
    trn_best_loss   = math.inf
    trn_best_rerror = math.inf

    for cepoch in range(0,numEpochs):
        print("Epoch :    ",str(cepoch))                
        ti = np.random.permutation(totalTrain)
        #np.random.shuffle(ti)
        
        model.train()
        with torch.enable_grad():                            
            # Bar
            bar = progressbar.ProgressBar(maxval=numTrain, widgets=[progressbar.Bar('=', '    trn[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            
            # Training
            loss_lst_epoch   = []
            rerror_lst_epoch = []
            for trn_idx in range(0, numTrain):
                bar.update(trn_idx+1)

                optimizer.zero_grad()
                X1 = XT[ ti[trn_idx] ]
                Y1 = YT[ ti[trn_idx] ]

                if False:
                    X11 = X1.detach().numpy()    
                    Y11 = Y1.detach().numpy()    
                    ScaleAndShow(X11[0].squeeze(), 1)
                    ScaleAndShow(Y11.squeeze(), 2)
                    plt.show()

                X1 = X1.unsqueeze(0)                
                X1 = X1.to(device)                         
                Y1 = Y1.squeeze()

                uOut = model(X1)
                loss, randError = dsn.RandLossDSN(uOut, Y1, lossType, trn_idx)

                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]
                #print("\t\tLoss   : " + str(loss.detach().numpy()) + "   Error: " + str(randError))
                # Don't change model on last pass so that train and validation are aligned
                if trn_idx < numTrain-1:
                    loss.backward()
                    optimizer.step()

            # Finish bar
            bar.finish()

            trn_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            trn_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
            trn_loss_lst       = trn_loss_lst   + [trn_cepoch_loss]
            trn_rerror_lst     = trn_rerror_lst + [trn_cepoch_rerror]
            

            trnLoss =  sum(trn_loss_lst) / len(trn_loss_lst)
            trnError =  sum(trn_rerror_lst) / len(trn_rerror_lst)

            print("   Loss: " + str(trn_cepoch_loss) + " Error: " + str(trn_cepoch_rerror))
            print("AVGLoss: " + str(trnLoss) + " Error: " + str(trnError))

            if (cepoch % logScores)==0:
                vname = "train1D_model_" + str(cepoch) + ".pth"
                torch.save(model.state_dict(), vname)            
                vname = "train1D_loss_" + str(cepoch) + ".pkl"                                
                with open(vname, 'wb') as f:
                    pickle.dump(trn_loss_lst, f)            
                vname = "train1D_error_" + str(cepoch) + ".pkl"                                
                with open(vname, 'wb') as f:
                    pickle.dump(trn_rerror_lst, f)            

    # Loading model
    # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
    # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
    # uOut1 = model(X3)
    # uOut2 = loaded_model(X3)
