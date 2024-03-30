import torch
import torch.nn.functional as F
from unet import UNet
import QuanSynData
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx
#import pdb
#pdb.set_trace()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())

Xsyn, Ysyn, XTsyn, YTsyn = QuanSynData.GetData(1)

### make ground truth binary for debugging
#Ysyn = (Ysyn != 0)
Ysyn = np.squeeze(Ysyn.astype(np.longlong))
if ( len(Ysyn.shape) < 3):
    Ysyn = np.expand_dims(Ysyn, axis=0)

YTsyn = np.squeeze(YTsyn.astype(np.longlong))
if ( len(YTsyn.shape) < 3):
    YTsyn = np.expand_dims(YTsyn, axis=0)

plt.figure(1)
plt.imshow(Xsyn[0,0])
plt.figure(2)
plt.imshow(Ysyn[0])
plt.figure(3)
plt.imshow(XTsyn[0,0])
plt.figure(4)
plt.imshow(YTsyn[0])

def GetConnectedComponentImage(xPred, yPred, threshold):

    G = nx.grid_2d_graph(xPred.shape[0], xPred.shape[1])
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:    #  vertical edges
            d['weight'] =  yPred[u[0], u[1]]
        else:               # horizontal edges
            d['weight'] =  xPred[u[0], u[1]]

    L = ev.GetLabelsBelowThreshold(G, threshold)
    img = np.zeros((xPred.shape[0], xPred.shape[1]), np.single)
    nlabel = dict()
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            img[i,j] = L[(i,j)]
            nlabel[L[(i,j)] ] = 1

    print('Unique labels: ' + str(len(nlabel)))
    return(img)


def GetRandWeights(xPred, yPred, nodeLabels):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    G = nx.grid_2d_graph(W, H) 
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:    #  vertical edges
            d['weight'] =  yPred[u[0], u[1]]
        else:               # horizontal edges
            d['weight'] =  xPred[u[0], u[1]]
                
        nlabels_dict[u] = nodeLabels[u[0], u[1]]
        nlabels_dict[v] = nodeLabels[v[0], v[1]]

    # Just use fixed threshold for now    
    bestT = 0.0
    [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict, True)
    
    # Inferning version for later
    #[bestT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
        
    # Now we assign MST edge errors back to UNET  
    xCounts = np.zeros((2, W, H), np.single)
    yCounts = np.zeros((2, W, H), np.single)
    threshPos = 0.0
    threshNeg = 0.0
        
    for i in range(len(posCounts)):
        (u,v) = mstEdges[i]
        edgeDiff = posCounts[i] - negCounts[i]
        edgeCount = posCounts[i] + negCounts[i]
        if edgeDiff > 0.0:      
            edgeSign = 1.0
            edgeWeight = edgeDiff / edgeCount
        else: 
            edgeSign = -1.0
            edgeWeight = -edgeDiff / edgeCount

        if u[0] == v[0]:    #  vertical edges
            yCounts[0, u[0], u[1]] = edgeSign                
            yCounts[1, u[0], u[1]] = edgeWeight
        else:               # horizontal edges
            xCounts[0, u[0], u[1]] = edgeSign
            xCounts[1, u[0], u[1]] = edgeWeight

        w = G.edges[u,v]['weight']
        if w > 0:
            threshPos = threshPos + posCounts[i] 
            threshNeg = threshNeg + negCounts[i] 

    posError = totalPos - threshPos
    negError = threshNeg
    randError = (posError + negError) / (totalPos + totalNeg)

                
    return [xCounts, yCounts, randError, totalPos, totalNeg]


def RandLossV1(uOut, yones, nodeLabels, epoch):

    # We have two outputs from the unet 
    # One corresponds to xedges
    # One corresponds to yedges
    # Code is particularly ugly since it handles them separately... 

    squOut = torch.squeeze(uOut)   
    xOut = squOut[0]
    yOut = squOut[1]
    
    npXPred = xOut.cpu().detach().numpy()
    npYPred = yOut.cpu().detach().numpy()
                    
    [xCounts, yCounts, randError, totalPos, totalNeg] = GetRandWeights(npXPred, npYPred, nodeLabels)
    #print('Total +ve weight: %f   -ve weight: %f' % (totalPos, totalNeg))
    xLabels = torch.tensor(xCounts[0])
    xWeights = torch.tensor(xCounts[1])
    yLabels = torch.tensor(yCounts[0])
    yWeights = torch.tensor(yCounts[1])
    
    # Squared loss just because it was the easiest thing for me to code without looking stuff up! 
    xErrors = (xOut.cpu() - xLabels) ** 2
    xCon = torch.mul(xErrors, xWeights)
    xloss = torch.sum(xCon)
    
    yErrors = (yOut.cpu() - yLabels) ** 2
    yCon = torch.mul(yErrors, yWeights)
    yloss = torch.sum(yCon)
    
    randLoss = xloss + yloss
    print("Epoch " + str(epoch) + ":   Loss " + str(randLoss.item()) + "  and Rand " + str(randError))
    return(randLoss)    

########################################################################################
########################################################################################
#### We only train with one image at a time since graphs are different for each image 
#### Graph defines the "minibatch"  

XS = np.expand_dims(Xsyn[0], axis=0)
YL = Ysyn[0]

X = torch.tensor(XS, requires_grad=True)
y = torch.ones([1, 2, XS.shape[2], XS.shape[3] ], dtype=torch.float32)

X = X.to(device) 

epochs = 1000

for epoch in range(epochs):

    uOut = model(X)  
        
    loss = RandLossV1(uOut, y, YL, epoch)    

    #myLoss = loss.item()
    #print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    
# Look at train image
uOut = model(X) 

squOut = torch.squeeze(uOut)   
xOut = squOut[0]
yOut = squOut[1]
    
npXPred = xOut.cpu().detach().numpy()
npYPred = yOut.cpu().detach().numpy()

img = GetConnectedComponentImage(npXPred, npYPred, 0.5)
plt.figure(5)
plt.imshow(img)

# And test
XTS = np.expand_dims(XTsyn[0], axis=0)
XT = torch.tensor(XTS)
XT = XT.to(device)  

uOut = model(XT) 

squOut = torch.squeeze(uOut)   
xOut = squOut[0]
yOut = squOut[1]
    
npXPred = xOut.cpu().detach().numpy()
npYPred = yOut.cpu().detach().numpy()

img = GetConnectedComponentImage(npXPred, npYPred, 0.5)
plt.figure(6)
plt.imshow(img)
plt.show()

exit()




# Saving a Model
#torch.save(model.state_dict(), MODEL_PATH)
# Loading the model.
#checkpoint = torch.load(MODEL_PATH)
#model.load_state_dict(checkpoint)
