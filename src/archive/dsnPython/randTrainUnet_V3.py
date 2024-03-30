import torch
import torch.nn.functional as F
from unet import UNet
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())

print("Loading")
(img, seg) = bsds.LoadTrain(0)    
(img1, seg1, img2, seg2) = bsds.ScaleAndCropData(img, seg)


bsds.VizTrainTest(img1, seg1, img2, seg2)
plt.figure(5)
plt.imshow(img1)
plt.ion()
plt.show()

Ysyn = np.expand_dims(seg1, axis=0)
YTsyn = np.expand_dims(seg2, axis=0)
Xsyn = np.zeros((img1.shape[2], img1.shape[0], img1.shape[1]))        
XTsyn = np.zeros((img2.shape[2], img2.shape[0], img2.shape[1]))        
for c in range(3):
    Xsyn[c,:,:] = img1[:,:,c]
    XTsyn[c,:,:] = img2[:,:,c]

Xsyn = np.expand_dims(Xsyn, axis=0)
XTsyn = np.expand_dims(XTsyn, axis=0)

Xsyn = Xsyn.astype(np.single)
XTsyn = XTsyn.astype(np.single)
Ysyn = Ysyn.astype(np.single)
YTsyn = YTsyn.astype(np.single)
#print('Shapes on input')
#print(Xsyn.shape)
#print(XTsyn.shape)
#print(Ysyn.shape)
#print(YTsyn.shape)

### make ground truth binary for debugging
#Ysyn = (Ysyn != 0)
#Ysyn = np.squeeze(Ysyn.astype(np.longlong))
#if ( len(Ysyn.shape) < 3):
#    Ysyn = np.expand_dims(Ysyn, axis=0)


def GetConnectedComponentImage(xPred, yPred, threshold):

    G = nx.grid_2d_graph(xPred.shape[0], xPred.shape[1])
    for u, v, d in G.edges(data = True):
        if u[0] == v[0]:    #  vertical edges
            d['weight'] =  yPred[u[0], u[1]]
        else:               # horizontal edges
            d['weight'] =  xPred[u[0], u[1]]

    L = ev.GetLabelsAtThreshold(G, threshold)
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
    [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
    
    # Inferning version for later
    #[bestT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
        
    # Now we assign MST edge errors back to UNET  
    xCounts = np.zeros((2, W, H), np.single)
    yCounts = np.zeros((2, W, H), np.single)
        
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
                
    return [xCounts, yCounts, bestT, totalPos, totalNeg]

def WatershedCut(uOut):
    # Modify the affinities
    squOut = torch.squeeze(uOut)   
    xyOut = squOut.cpu().detach().numpy()
    
    #xleft  = [ npX[j,i-1], npY[j-1,i], npY[j,i] ]
    #xright = [ npX[j,i+1], npY[j-1,i+1], npY[j,i+1] ]
    #yleft  = [ npY[j-1,i], npX[j,i-1], npX[j,i] ]
    #yright - [ npY[j+1,i], npX[j+1,i-1], npX[j+1,i]]
    minMaxIndicies = np.zeros((2, xyOut.shape[1]-1, xyOut.shape[2]-1, 3), np.long)
    # Neighbor indices (#edge_planes, #neighor_sets, #neighbors, #dim_index)
    ni = np.zeros((2, 2, 3, 3), np.long)
    # X edges (left and right neighbor sets)
    ni[0, 0] = [ [0,  0, -1], [1, -1,  0], [1, 0, 0] ]
    ni[0, 1] = [ [0,  0,  1], [1, -1,  1], [1, 0, 1] ]
    # Y edges (top and bottom neighbor sets)
    ni[1, 0] = [ [1, -1,  0], [0,  0, -1], [0, 0, 0] ]
    ni[1, 1] = [ [1,  1,  0], [0,  1, -1], [0, 1, 0] ]

    for j in range(0, xyOut.shape[1]-1):
        for i in range(0, xyOut.shape[2]-1):
            for p in range(0, ni.shape[0]):
                minVal = 1.0e12
                minInd = [-1, -1, -1]
                for s in range(0, ni.shape[1]):
                    maxVal = -1.0e12
                    maxInd = [-1, -1, -1]
                    for n in range(0, ni.shape[2]):
                        npp = ni[p, s, n, 0]
                        njj = j + ni[p, s, n, 1]
                        nii = i + ni[p, s, n, 2]
                        if (njj >= 0) and (nii >= 0) and (njj <= xyOut.shape[1]-2) and (nii <= xyOut.shape[2]-2):
                            if xyOut[npp, njj, nii] > maxVal:
                                maxVal = xyOut[npp, njj, nii]
                                maxInd = [npp, njj, nii]
                    
                    if maxVal < minVal:
                        minVal = maxVal 
                        minInd = maxInd

                # Now we have maxmin edge index we can calculate the watershed edge cut in pytorch
                squOut[p, j, i] = squOut[p, j, i] - squOut[minInd[0], minInd[1], minInd[2]] 
                #squOut[p, j, i] = squOut[minInd[0], minInd[1], minInd[2]] - squOut[p, j, i]

    return squOut


def RandLossV2(uOut, yones, nodeLabels):

    # We have two outputs from the unet 
    # One corresponds to xedges
    # One corresponds to yedges
    # Code is particularly ugly since it handles them separately... 

    squOut = torch.squeeze(uOut)   
    xOut = squOut[0]
    yOut = squOut[1]
    
    npXPred = xOut.cpu().detach().numpy()
    npYPred = yOut.cpu().detach().numpy()
                    
    [xCounts, yCounts, bestT, totalPos, totalNeg] = GetRandWeights(npXPred, npYPred, nodeLabels)
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
epochPlot = 10

for epoch in range(epochs):

    uOut = model(X)  
    
    wsOut = WatershedCut(uOut)

    loss = RandLossV2(wsOut, y, YL)    

    myLoss = loss.item()
    print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (epoch % epochPlot) == 0:
        squOut = torch.squeeze(wsOut)   
        xOut = squOut[0]
        yOut = squOut[1]
            
        npXPred = xOut.cpu().detach().numpy()
        npYPred = yOut.cpu().detach().numpy()

        img = GetConnectedComponentImage(npXPred, npYPred, 0.0)
        
        plt.imshow(img)
        plt.pause(0.0001)        

    
    
# Look at train image
uOut = model(X) 
wsOut = WatershedCut(uOut)
squOut = torch.squeeze(wsOut)   
xOut = squOut[0]
yOut = squOut[1]
    
npXPred = xOut.cpu().detach().numpy()
npYPred = yOut.cpu().detach().numpy()

img = GetConnectedComponentImage(npXPred, npYPred, 0.0)
plt.figure(5)
plt.imshow(img)

# And test
XTS = np.expand_dims(XTsyn[0], axis=0)
XT = torch.tensor(XTS)
XT = XT.to(device)  

uOut = model(XT) 
wsOut = WatershedCut(uOut)
squOut = torch.squeeze(wsOut)   
xOut = squOut[0]
yOut = squOut[1]
    
npXPred = xOut.cpu().detach().numpy()
npYPred = yOut.cpu().detach().numpy()

img = GetConnectedComponentImage(npXPred, npYPred, 0.0)
plt.figure(6)
plt.imshow(img)
plt.show()

exit()




# Saving a Model
#torch.save(model.state_dict(), MODEL_PATH)
# Loading the model.
#checkpoint = torch.load(MODEL_PATH)
#model.load_state_dict(checkpoint)
