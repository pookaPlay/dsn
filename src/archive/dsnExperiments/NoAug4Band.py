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

    numEpochs = 1000
    numTrain = 50
    numValid = 50  
    logScores = 10
    unetFeatures = 6
    unetDepth = 5
    lossType = 'error'
    saveNum = lossType
    useBatchNorm = True

    learningRate = 1
    rhoMemory = 0.99
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

    #Ysyn = Ysyn.astype(np.single)

    XT = torch.tensor(X, requires_grad=False)                
    YT = torch.tensor(Y, requires_grad=False)                                                
    XVT = torch.tensor(XV, requires_grad=False)                
    YVT = torch.tensor(YV, requires_grad=False)                                                

    totalTrain = XT.shape[0]
    totalValid = XVT.shape[0]
    if numTrain < 0:
        numTrain = totalTrain
    if numValid < 0:
        numValid = totalValid

    print("Batch: " + str(numTrain) + " out of " + str(totalTrain) + " and " + str(numValid) + " out of " + str(totalValid))

    # Setting up U-net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNet(in_channels=4, n_classes=1, depth=unetDepth, wf=unetFeatures, padding=True, batch_norm=useBatchNorm, up_mode='upsample').to(device)

    optimizer = optim.Adadelta(model.parameters(), rho=rhoMemory, lr=learningRate)
    
    # Initializing training and validation lists to be empty
    trn_loss_lst   = []
    trn_rerror_lst = []
    val_loss_lst   = []
    val_rerror_lst  = []

    # Initializing best loss and random error
    val_best_loss   = math.inf
    val_best_rerror = math.inf
    trn_best_loss   = math.inf
    trn_best_rerror = math.inf

    tia = np.random.permutation(totalTrain)
    vi = np.random.permutation(totalValid)

    for cepoch in range(0,numEpochs):
        print("Epoch :    ",str(cepoch))                
        ti = tia[0:numTrain]
        np.random.shuffle(ti)
        
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
                    ScaleAndShow(X11[1].squeeze(), 2)
                    ScaleAndShow(X11[2].squeeze(), 3)
                    ScaleAndShow(X11[3].squeeze(), 4)
                    ScaleAndShow(Y11.squeeze(), 5)
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
            trn_rerror_lst     = trn_rerror_lst + [trn_cepoch_rerror]
            trn_loss_lst       = trn_loss_lst   + [trn_cepoch_loss]
    
            if trn_cepoch_loss < trn_best_loss:
                trn_best_loss       = trn_cepoch_loss
                trn_best_loss_epoch = cepoch
                trn_best_loss_model = model

            if trn_cepoch_rerror < trn_best_rerror:
                trn_best_rerror       = trn_cepoch_rerror
                trn_best_rerror_epoch = cepoch
                trn_best_rerror_model = model
                fname = "train_error_model_" + str(saveNum) + ".pth"
                print("\t\tSaving training model with error " + str(trn_best_rerror) + " to " + fname)                
                torch.save(trn_best_rerror_model.state_dict(), fname)

            print("AVG Loss   : " + str(trn_cepoch_loss) + "   Err: " + str(trn_cepoch_rerror))
            print("BEST Loss  : " + str(trn_best_loss) + "   Err: " + str(trn_best_rerror) + " at " + str(trn_best_loss_epoch) + ", " + str(trn_best_rerror_epoch))
            
            # Saving all the losses and rand_errors
            if ((cepoch % logScores) == 0): 
                print("Saving train loss and errors")
                vname = "train_loss_" + str(saveNum) + "_" + str(cepoch) + ".pkl"
                with open(vname, 'wb') as f:
                    pickle.dump(trn_loss_lst, f)
                vname = "train_error_" + str(saveNum) + "_" + str(cepoch) + ".pkl"
                with open(vname, 'wb') as f:
                    pickle.dump(trn_rerror_lst, f)            

        # Validation every epoch        
        loss_lst_epoch   = []
        rerror_lst_epoch = []
        # Bar
        model.eval()
        with torch.no_grad():                

            bar = progressbar.ProgressBar(maxval=numValid, widgets=[progressbar.Bar('-', '    Val[', ']'), ' ', progressbar.Percentage()])
            bar.start()            

            for val_idx in range(0, numValid):
                bar.update(val_idx+1)
                # print("\t Validating on ",str(val_idx)," image")

                XV1 = XVT[ vi[val_idx] ]
                YV1 = YVT[ vi[val_idx] ]

                if False:
                    X11 = XV1.detach().numpy()    
                    Y11 = YV1.detach().numpy()    
                    ScaleAndShow(X11[0].squeeze(), 1)
                    ScaleAndShow(X11[1].squeeze(), 2)
                    ScaleAndShow(X11[2].squeeze(), 3)
                    ScaleAndShow(X11[3].squeeze(), 4)
                    ScaleAndShow(Y11.squeeze(), 5)
                    plt.show()

                XV1 = XV1.unsqueeze(0)
                XV1 = XV1.to(device)                         
                YV1 = YV1.squeeze()

                uOut = model(XV1)
                
                loss, randError = dsn.RandLossDSN(uOut, YV1, lossType, cepoch)

                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]

            # Finish bar
            bar.finish()            

            val_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            val_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
            val_rerror_lst     = val_rerror_lst + [val_cepoch_rerror]
            val_loss_lst       = val_loss_lst   + [val_cepoch_loss]

            if val_cepoch_loss < val_best_loss:
                # Saving best loss model
                val_best_loss       = val_cepoch_loss
                val_best_loss_epoch = cepoch
                val_best_loss_model = model
                #print("\t\tBest val loss ", str(val_best_loss))
                #print("\t\tCurrent val error ", str(val_best_rerror))
                #torch.save(val_best_loss_model.state_dict(), "val_loss_model_1.pth")

            if val_cepoch_rerror < val_best_rerror:
                # Saving best rerror model
                val_best_rerror       = val_cepoch_rerror
                val_best_rerror_epoch = cepoch
                val_best_rerror_model = model
                fname = "valid_error_model_" + str(saveNum) + ".pth"
                print("\t\tSaving valid model with error " + str(val_best_rerror) + " to " + fname)                
                torch.save(val_best_rerror_model.state_dict(), fname)

            print("AVG Valid   : " + str(val_cepoch_loss) + "   Err: " + str(val_cepoch_rerror))
            print("BEST Valid  : " + str(val_best_loss) + "   Err: " + str(val_best_rerror) + " at " + str(val_best_loss_epoch) + ", " + str(val_best_rerror_epoch))

            if ((cepoch % logScores) == 0): 
                print("Saving valid loss and error")
                vname = "valid_loss_" + str(saveNum) + "_" + str(cepoch) + ".pkl"                
                with open(vname, 'wb') as f:
                    pickle.dump(val_loss_lst, f)
                vname = "valid_error_" + str(saveNum) + "_" + str(cepoch) + ".pkl"                                
                with open(vname, 'wb') as f:
                    pickle.dump(val_rerror_lst, f)            

    
    # Loading model
    # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
    # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
    # uOut1 = model(X3)
    # uOut2 = loaded_model(X3)
