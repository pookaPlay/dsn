import torch
import torch.nn.functional as F
from unet import UNet
import QuanSynData
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# was
optimizer = torch.optim.Adam(model.parameters())
verbose = 1

def GetLabelsAtThreshold(G, threshold):
    lg = G.copy()    
    lg.remove_edges_from([(u,v) for (u,v,d) in  G.edges(data=True) if d['weight'] > threshold])
    L = {node:color for color,comp in enumerate(nx.connected_components(lg)) for node in comp}    
    return L

def GetCCImage(G, threshold, W, H):
    L = GetLabelsAtThreshold(G, threshold)
    img = np.zeros((W, H), np.single)
    nlabel = dict()
    for i in range(img.shape[0]):        
        for j in range(img.shape[1]):
            img[i,j] = L[(i,j)]
            nlabel[L[(i,j)] ] = 1

    print('Unique labels: ' + str(len(nlabel)))
    return(img)

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')

def GetRandWeights(G, nodeLabels):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        nlabels_dict[u] = nodeLabels[u[0], u[1]]
        nlabels_dict[v] = nodeLabels[v[0], v[1]]

    # Just use fixed threshold for now    
    bestT = 0.0
    [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict, False)
    
    # Inferning version for later
    #[bestT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
        
    # Now we assign MST edge errors back to UNET  
    eCounts = np.zeros((2, W, H), np.single)    
        
    for i in range(len(posCounts)):
        (u,v) = mstEdges[i]
        edgeDiff = posCounts[i] - negCounts[i]
        edgeCount = posCounts[i] + negCounts[i]
        if edgeDiff > 0.0:      
            edgeSign = -1.0
            edgeWeight = edgeDiff / edgeCount
        else: 
            edgeSign = 1.0
            edgeWeight = -edgeDiff / edgeCount

        
        eCounts[0, u[0], u[1]] = edgeSign                
        eCounts[1, u[0], u[1]] = edgeWeight
                
    return [eCounts, bestT, totalPos, totalNeg]

def RandLossV2(uOut, nodeLabels, G):

    xyOut = torch.squeeze(uOut)   
    #xyOut = squOut[0]        
                    
    [eCounts, bestT, totalPos, totalNeg] = GetRandWeights(G, nodeLabels)
    #print('Total +ve weight: %f   -ve weight: %f' % (totalPos, totalNeg))
    eLabels = torch.tensor(eCounts[0])
    eWeights = torch.tensor(eCounts[1])
    
    # Squared loss just because it was the easiest thing for me to code without looking stuff up! 
    eErrors = (xyOut.cpu() - eLabels) ** 2
    eCon = torch.mul(eErrors, eWeights)
    randLoss = torch.sum(eCon)
    
    return(randLoss)    

def GetWatershedCut(GXY, W, H):
    G = nx.grid_2d_graph(W, H) 
    print('WatershedCut')
    print(GXY.shape)
    
    for u, v, d in G.edges(data = True):        
        d['weight'] =  GXY[u[0], u[1]] + GXY[v[0], v[1]]
       
    #PlotSubGraph(G, 7, 5, 9)

    #this function returns the graph WG with new weights of max(min(neighbors))    
    for (u,v,d) in G.edges(data = True):
        #print(u)        
        uev = [ues for ues in G[u] if ues != v]
        veu = [ves for ves in G[v] if ves != u]

        uew = [G[u][ues]['weight'] for ues in uev]
        maxUW = min(uew)
        maxUI = uew.index(maxUW)
        maxUV = uev[maxUI]  # should be vertix v of edge uv that had max

        vew = [G[v][ves]['weight'] for ves in veu]
        maxVW = min(vew)
        maxVI = vew.index(maxVW)
        maxVU = veu[maxVI]  # should be vertex u of edge vu that had max

        # now do max
        if maxUW >= maxVW:
            minMaxW = maxUW
            minMaxU = u
            minMaxV = maxUV
        else:
            minMaxW = maxVW
            minMaxU = v
            minMaxV = maxVU        

        d['weight'] = d['weight'] - minMaxW
        d['minmax'] = (minMaxU, minMaxV)
        
    #PlotSubGraph(G, 7, 5, 9)
    #imgX[8, 14] = 100.0 
    #imgX[23, 24] = 100.0
    return G

def ApplyWatershedCut(G, uOut):
    
    wsOut = torch.zeros([1, 1, uOut.shape[2], uOut.shape[3] ], dtype=torch.float32)
    
    for (u,v,d) in G.edges(data = True):

        wsOut[0,0,u[0],u[1]] = uOut[0,0,u[0], u[1]] + uOut[0,0,v[0], v[1]] - uOut[0,0,d['minmax'][0][0], d['minmax'][0][1] ] - uOut[0,0,d['minmax'][1][0], d['minmax'][1][1] ]

    return(wsOut)

#i = Variable(torch.from_numpy(indexes).long())
#respos = torch.index_select(flat, 1, i)
def DirGrad(img):
    
    left = img
    right = F.pad(img, [0, 1, 0, 0])[:, :, :, 1:]
    top = img
    bottom = F.pad(img, [0, 0, 0, 1])[:, :, 1:, :]
    dx = right - left
    dx[:, :, :, -1] = 0
    #dxx = torch.pow(dx, 2) 
    dy = bottom - top
    dy[:, :, -1, :] = 0
    #dyy = torch.pow(dy, 2)
    DXY = torch.sqrt(torch.pow(dx,2)+ torch.pow(dy,2))
    #DXY = torch.sqrt(torch.abs(dx)+ torch.abs(dy))            
    GXY = F.max_pool2d(DXY, (2,2))
    #dyout = F.max_pool2d(dyy, (2,2))
    #print('DirGrad sizes')
    #print(dxout.shape)
    #print(dyout.shape)
    #G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return GXY

########################################################################################
########################################################################################
#### We only train with one image at a time since graphs are different for each image 
#### Graph defines the "minibatch"  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=1, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())

print("Loading")
Xsyn, Ysyn, XTsyn, YTsyn = QuanSynData.GetData(1, 64, 2)
Xsyn2, Ysyn2, XTsyn2, YTsyn2 = QuanSynData.GetData(1, 64, 4)

### make ground truth binary for debugging
#Ysyn = (Ysyn != 0)
Ysyn = np.squeeze(Ysyn.astype(np.longlong))
if ( len(Ysyn.shape) < 3):
    Ysyn = np.expand_dims(Ysyn, axis=0)

Ysyn2 = np.squeeze(Ysyn2.astype(np.longlong))
if ( len(Ysyn2.shape) < 3):
    Ysyn2 = np.expand_dims(Ysyn2, axis=0)

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
plt.ion()
plt.show()

#Xsyn = np.expand_dims(Xsyn, axis=0)
#XTsyn = np.expand_dims(XTsyn, axis=0)

#Xsyn = Xsyn.astype(np.single)
#XTsyn = XTsyn.astype(np.single)
#Ysyn = Ysyn.astype(np.single)
#YTsyn = YTsyn.astype(np.single)
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

XS = np.expand_dims(Xsyn[0], axis=0)
W = XS.shape[2]
H = XS.shape[3]
W = int(W/2)
H = int(H/2)
YL = Ysyn2[0]

X = torch.tensor(XS, requires_grad=True)
y = torch.ones([1, 2, XS.shape[2], XS.shape[3] ], dtype=torch.float32)

X = X.to(device) 

epochs = 1000
epochPlot = 10
threshold = 0.0

for epoch in range(epochs):

    dOut = model(X)  
    
    uOut = DirGrad(dOut)

    uOut1 = torch.squeeze(uOut)   
    npuOut = uOut1.cpu().detach().numpy()

    G = GetWatershedCut(npuOut, W, H)
    wsOut = ApplyWatershedCut(G, uOut)

    loss = RandLossV2(wsOut, YL, G)    

    myLoss = loss.item()
    print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))

    optim.zero_grad()
    loss.backward()
    optim.step()

    if 1: #(epoch % epochPlot) == 0:
        img = GetCCImage(G, threshold, W, H)       
        plt.imshow(img)
        plt.pause(0.0001)        

    
plt.waitforbuttonpress()
exit()
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
