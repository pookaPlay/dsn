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
model = UNet(in_channels=3, n_classes=1, depth=3, padding=True, up_mode='upsample').to(device)
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
#plt.figure(3)
#plt.imshow(XTsyn[0,0])
#plt.figure(4)
#plt.imshow(YTsyn[0])

def GetConnectedComponentImage(xyPred, threshold):

    G = nx.grid_2d_graph(xPred.shape[0], xPred.shape[1])
    for u, v, d in G.edges(data = True):
        d['weight'] =  (xyPred[u[0], u[1]] + xyPred[v[0], v[1]])/2.0

    L = ev.GetLabelsBelowThreshold(G, threshold)
    img = np.zeros((xyPred.shape[0], xyPred.shape[1]), np.single)
    nlabel = dict()
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            img[i,j] = L[(i,j)]
            nlabel[L[(i,j)] ] = 1

    print('Unique labels: ' + str(len(nlabel)))
    return(img)


def GetRandWeights(xyPred, nodeLabels):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    G = nx.grid_2d_graph(W, H) 
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        d['weight'] =  (xyPred[u[0], u[1]] + xyPred[v[0], v[1]])/2.0                
        nlabels_dict[u] = nodeLabels[u[0], u[1]]
        nlabels_dict[v] = nodeLabels[v[0], v[1]]

    # Just use fixed threshold for now    
    bestT = 0.0
    [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict, True)
    # Inferning version for later
    #[bestT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
        
    # Now we assign MST edge errors back to UNET  
    xyCounts = np.zeros((2, W, H), np.single)
    xyLabelWeights = np.zeros((2, W, H), np.single)
    threshPos = 0.0
    threshNeg = 0.0

    for i in range(len(posCounts)):
        (u,v) = mstEdges[i]
        
        xyCounts[0, u[0], u[1]] = xyCounts[0, u[0], u[1]] + posCounts[i]
        xyCounts[1, u[0], u[1]] = xyCounts[1, u[0], u[1]] + negCounts[i]
        xyCounts[0, v[0], v[1]] = xyCounts[0, v[0], v[1]] + posCounts[i]
        xyCounts[1, v[0], v[1]] = xyCounts[1, v[0], v[1]] + negCounts[i]

        w = G.edges[u,v]['weight']
        if w > 0:
            threshPos = threshPos + posCounts[i] 
            threshNeg = threshNeg + negCounts[i] 

    posError = totalPos - threshPos
    negError = threshNeg
    randError = (posError + negError) / (totalPos + totalNeg)
    #print("-----------------------------"); 
    #print("Rand Error: " + str(randError))
    #print("From #pos: " + str(totalPos) + " #neg: " + str(totalNeg))
    #print("   and FN: " + str(posError) + "   FP: " + str(negError))        


    for i in range(len(posCounts)):
        (u,v) = mstEdges[i]

        w = u
        edgeDiff = xyCounts[0, w[0], w[1]] - xyCounts[1, w[0], w[1]]
        edgeCount = xyCounts[0, w[0], w[1]] + xyCounts[1, w[0], w[1]]
        if edgeDiff > 0.0:      
            edgeSign = 1.0
            edgeWeight = edgeDiff / edgeCount
        else: 
            edgeSign = -1.0
            edgeWeight = -edgeDiff / edgeCount
        xyLabelWeights[0, w[0], w[1]] = edgeSign
        xyLabelWeights[1, w[0], w[1]] = edgeWeight

        w = v
        edgeDiff = xyCounts[0, w[0], w[1]] - xyCounts[1, w[0], w[1]]
        edgeCount = xyCounts[0, w[0], w[1]] + xyCounts[1, w[0], w[1]]
        if edgeDiff > 0.0:      
            edgeSign = 1.0
            edgeWeight = edgeDiff / edgeCount
        else: 
            edgeSign = -1.0
            edgeWeight = -edgeDiff / edgeCount
        xyLabelWeights[0, w[0], w[1]] = edgeSign
        xyLabelWeights[1, w[0], w[1]] = edgeWeight

    #print(xyCounts)    

    return [xyLabelWeights, randError, totalPos, totalNeg]


def RandLossV1(uOut, yones, nodeLabels, epoch):

    # We have two outputs from the unet 
    # One corresponds to xedges
    # One corresponds to yedges
    # Code is particularly ugly since it handles them separately... 

    xyOut = torch.squeeze(uOut)   
    xyOut = xyOut.cpu()    
    npXYPred = xyOut.detach().numpy()
                        
    [xyCounts, randError, totalPos, totalNeg] = GetRandWeights(npXYPred, nodeLabels)
    #print('Total +ve weight: %f   -ve weight: %f' % (totalPos, totalNeg))
    xyLabels = torch.tensor(xyCounts[0])
    xyWeights = torch.tensor(xyCounts[1])
    
    # Squared loss just because it was the easiest thing for me to code without looking stuff up! 
    xyErrors = (xyOut - xyLabels) ** 2
    xyCon = torch.mul(xyErrors, xyWeights)
    xyloss = torch.sum(xyCon)
    
    print("Epoch " + str(epoch) + ":   Loss " + str(xyloss.item()) + "  and Rand " + str(randError))
    return(xyloss)    


def Sobel(img):
    #Black and white input image x, 1x1xHxW
    a = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    b = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)    
    c = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    #d = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    a = a.to(device) 
    b = b.to(device) 
    c = c.to(device) 
    #d = d.to(device) 

    a = a.view((1,1,3,3))
    b = b.view((1,1,3,3))
    c = c.view((1,1,3,3))
    #d = d.view((1,1,5,5))
    #print('In sobel type')
    #print(imgIn.type())
    #print(a.type())
    
    imgIn = F.conv2d(img, c, padding=(1,1))
    #imgIn = F.conv2d(img, d, padding=(2,2))
    #imgIn = img
    G_x = F.conv2d(imgIn, a, padding=(1,1))
    G_y = F.conv2d(imgIn, b, padding=(1,1))

    G_x = torch.pow(G_x,2) 
    G_y = torch.pow(G_y,2) 
    GXY = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    #DXY = F.max_pool2d(GXY, (2,2))
    #DY = F.max_pool2d(GY, (2,2))    
    return G_x, G_y, GXY

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
    #[G_x, G_y, GXY] = Sobel(uOut)
        
    loss = RandLossV1(uOut, y, YL, epoch)    

    #myLoss = loss.item()
    #print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))

    optim.zero_grad()
    loss.backward()
    optim.step()
    
    
# Look at train image
uOut = model(X) 

xyOut = torch.squeeze(uOut)      
npXYPred = xyOut.cpu().detach().numpy()

img = GetConnectedComponentImage(npXYPred, 0.5)
plt.figure(5)
plt.imshow(img)
plt.show()

exit()




# Saving a Model
#torch.save(model.state_dict(), MODEL_PATH)
# Loading the model.
#checkpoint = torch.load(MODEL_PATH)
#model.load_state_dict(checkpoint)
