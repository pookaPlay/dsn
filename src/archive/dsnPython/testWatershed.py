import torch
import torch.nn.functional as F
from unet import UNet
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx
import SynGraph as syng
import VizGraph as vizg

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

def PlotSubGraph(G, subSize, subx1, suby1):
    A = dict() 
    subx2 = subx1 + subSize - 1
    suby2 = suby1 + subSize - 1
    for (u,v,d) in G.edges(data = True):

        if u[0] >= subx1 and u[0] <= subx2:
            if u[1] >= suby1 and u[1] <= suby2:
                if v[0] >= subx1 and v[0] <= subx2:
                    if v[1] >= suby1 and v[1] <= suby2:
                        uu = (u[0] - subx1, u[1] - suby1) 
                        vv = (v[0] - subx1, v[1] - suby1)
                        if uu not in A:
                            A[uu] = dict()
                        if vv not in A:
                            A[vv] = dict()

                        A[uu][vv] = d['weight']
                        A[vv][uu] = d['weight']
        #if u[0] == 7 and u[1] == 15 and v[0]==8 and v[1] == 15:
        #    print("(7,15) -> (8,15) is: %f" % d['weight'] )
    GA = syng.InitWithAffinities(subSize, subSize, A) 
    vizg.DrawGraph(GA) #, labels=None, title=None, figSize=None, nodeSize=None):

# def GetWatershedCut(GX, GY, W, H):
#     G = nx.grid_2d_graph(W, H) 
#     print('WatershedCut')
#     print(GX.shape)
#     print(GY.shape)

#     for u, v, d in G.edges(data = True):
#         if u[0] == v[0]:    #  vertical edges
#             d['weight'] =  GY[u[0], u[1] + 1]
#         else:               # horizontal edges
#             d['weight'] =  GX[u[0]+1, u[1]]
        
#     #PlotSubGraph(G, 7, 5, 9)

#     imgX = np.zeros((H, W))
#     imgY = np.zeros((H, W))

#     #this function returns the graph WG with new weights of max(min(neighbors))    
#     for (u,v,d) in G.edges(data = True):
#         #print(u)        
#         uev = [ues for ues in G[u] if ues != v]
#         veu = [ves for ves in G[v] if ves != u]

#         uew = [G[u][ues]['weight'] for ues in uev]
#         maxUW = min(uew)
#         maxUI = uew.index(maxUW)
#         maxUV = uev[maxUI]  # should be vertix v of edge uv that had max

#         vew = [G[v][ves]['weight'] for ves in veu]
#         maxVW = min(vew)
#         maxVI = vew.index(maxVW)
#         maxVU = veu[maxVI]  # should be vertex u of edge vu that had max

#         # now do max
#         if maxUW >= maxVW:
#             minMaxW = maxUW
#             minMaxU = u
#             minMaxV = maxUV
#         else:
#             minMaxW = maxVW
#             minMaxU = v
#             minMaxV = maxVU        

#         d['weight'] = d['weight'] - minMaxW
#         d['minmax'] = (minMaxU, minMaxV)
#         if u[0] == v[0]:    #  vertical edges
#             imgY[u[0], u[1]] = d['weight'] 
#         else:               # horizontal edges
#             imgX[u[0], u[1]] = d['weight']             
        
#     #PlotSubGraph(G, 7, 5, 9)
#     #imgX[8, 14] = 100.0 
#     #imgX[23, 24] = 100.0
#     return G, imgX, imgY

def GetWatershedCut(GXY, W, H):
    G = nx.grid_2d_graph(W, H) 
    print('WatershedCut')
    print(GXY.shape)
    
    for u, v, d in G.edges(data = True):        
        d['weight'] =  GXY[u[0], u[1]] + GXY[v[0], v[1]]
       
    PlotSubGraph(G, 7, 20, 20)

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
        
    PlotSubGraph(G, 7, 20, 20)
    #imgX[8, 14] = 100.0 
    #imgX[23, 24] = 100.0
    return G

########################################################################################
########################################################################################
#### Feature Transforms

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

def DirAvg(img):
    
    left = img
    right = F.pad(img, [0, 1, 0, 0])[:, :, :, 1:]
    top = img
    bottom = F.pad(img, [0, 0, 0, 1])[:, :, 1:, :]
    dx = right + left
    dx = dx / 2.0
    dy = bottom + top    
    dy = dy / 2.0
    #G = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return dx, dy

def Avg(img):
    #Black and white input image x, 1x1xHxW
    c = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    d = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    c = c.to(device) 
    d = d.to(device) 

    c = c.view((1,1,3,3))
    d = d.view((1,1,5,5))
    
    imgIn = F.conv2d(img, c, padding=(1,1))
    #imgIn = F.conv2d(img, d, padding=(2,2))
    #imgIn = img
    return imgIn

def Sobel(img):
    #Black and white input image x, 1x1xHxW
    a = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
    b = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)    
    c = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32)
    d = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32)
    a = a.to(device) 
    b = b.to(device) 
    c = c.to(device) 
    d = d.to(device) 

    a = a.view((1,1,3,3))
    b = b.view((1,1,3,3))
    c = c.view((1,1,3,3))
    d = d.view((1,1,5,5))
    #print('In sobel type')
    #print(imgIn.type())
    #print(a.type())
    
    #imgIn = F.conv2d(img, c, padding=(1,1))
    #imgIn = F.conv2d(img, d, padding=(2,2))
    imgIn = img
    G_x = F.conv2d(imgIn, a, padding=(1,1))
    G_y = F.conv2d(imgIn, b, padding=(1,1))

    G_x = torch.pow(G_x,2) 
    G_y = torch.pow(G_y,2) 
    GXY = torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    #DXY = F.max_pool2d(GXY, (2,2))
    #DY = F.max_pool2d(GY, (2,2))    
    return GXY

########################################################################################
########################################################################################
#### We only train with one image at a time since graphs are different for each image 
#### Graph defines the "minibatch"  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())

print("Loading")
(img, seg) = bsds.LoadTrain(0)    
(img1, seg1, img2, seg2) = bsds.ScaleAndCropData(img, seg)

img1 = np.zeros((64, 64, 3))        
rimg = np.random.random_sample([64,64]) * 0.1 - 0.05
for j in range(64):
    for i in range(64):
        newVal = (i*128.0)/63.0 + (j*127.0)/63.0
        
        img1[i,j,0] = newVal + rimg[i,j]
        img1[i,j,1] = newVal + rimg[i,j]
        img1[i,j,2] = newVal + rimg[i,j]

#val = np.subtract(img1[20:30,20:30,0], 1000.0)
#maxv = np.max(val)
#minv = np.min(val)
#print('Val Max ' + str(maxv) + ' Min: ' + str(minv))

img1[10:20,20:30,:] = np.add(img1[10:20,20:30,:], 50.0)
img1[40:50,40:50,:] = np.add(img1[40:50,40:50,:], 50.0)


maxv1 = np.max(img1[10:20,20:30,:] )
minv1 = np.min(img1[10:20,20:30,:] )
maxv2 = np.max(img1[40:50,40:50,:] )
minv2 = np.min(img1[40:50,40:50,:] )
print('Obj 1 Max ' + str(maxv1) + ' Min: ' + str(minv1))
print('Obj 2 Max ' + str(maxv2) + ' Min: ' + str(minv2))
maxv = np.max(img1)
minv = np.min(img1)
print('Max ' + str(maxv) + ' Min: ' + str(minv))

img1  = img1 / (256.0)

#bsds.VizTrainTest(img1, seg1, img2, seg2)
#plt.figure(3)
#plt.imshow(img1)
#plt.ion()
#plt.show()

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
XS = np.expand_dims(Xsyn[0], axis=0)
YL = Ysyn[0]
W = XS.shape[2]
H = XS.shape[3]
W2 = int(W/2)
H2 = int(H/2)
print('Width: ' + str(W) + ' Height: ' + str(H))

X = torch.tensor(XS, requires_grad=True)
y = torch.ones([1, 2, XS.shape[2], XS.shape[3] ], dtype=torch.float32)

X = X.to(device) 
print('Into device')
print(X.shape)
XP = X[0,0]
XPP = XP[None, None, :, :]

npXP = XP.cpu().detach().numpy()
ScaleAndShow(npXP, 1)
#print(XPP.shape)
#print(XPP.type())
#GX, GY = Sobel(XPP)
#GXY = Sobel(XPP)
#XPPS = Avg(XPP)
GXY = DirGrad(XPP)

GXY1 = torch.squeeze(GXY)   
npGXY = GXY1.cpu().detach().numpy()
#GX1 = torch.squeeze(GX)   
#npGX = GX1.cpu().detach().numpy()
#GY1 = torch.squeeze(GY)   
#npGY = GY1.cpu().detach().numpy()

#print(npGX.shape)
#print(npGY.shape)
ScaleAndShow(npGXY, 5)
#ScaleAndShow(npGX, 5)
#ScaleAndShow(npGY, 6)
#PlotSubGraph(G, 7, 5, 9)
#G, imgX, imgY = GetWatershedCut(npGX, npGY, W, H)
G = GetWatershedCut(npGXY, W2, H2)

#ScaleAndShow(imgX, 7)
#ScaleAndShow(imgY, 8)



img = GetCCImage(G, 0.1, W2, H2)
ScaleAndShow(img, 4)

plt.show()

#epochs = 1000
#epochPlot = 10
## for epoch in range(epochs):

#     uOut = model(X)  
    
#     gOut = GraphCut(uOut)

#     loss = RandLossV2(gOut, y, YL)    

#     myLoss = loss.item()
#     print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))

#     optim.zero_grad()
#     loss.backward()
#     optim.step()

#     if (epoch % epochPlot) == 0:
#         squOut = torch.squeeze(gOut)   
#         xOut = squOut[0]
#         yOut = squOut[1]
            
#         npXPred = xOut.cpu().detach().numpy()
#         npYPred = yOut.cpu().detach().numpy()

#         img = GetConnectedComponentImage(npXPred, npYPred, 0.0)
        
#         plt.imshow(img)
#         plt.pause(0.0001)        

    
    
# # Look at train image
# uOut = model(X) 
# wsOut = WatershedCut(uOut)
# squOut = torch.squeeze(wsOut)   
# xOut = squOut[0]
# yOut = squOut[1]
    
# npXPred = xOut.cpu().detach().numpy()
# npYPred = yOut.cpu().detach().numpy()

# img = GetConnectedComponentImage(npXPred, npYPred, 0.0)
# plt.figure(5)
# plt.imshow(img)

# # And test
# XTS = np.expand_dims(XTsyn[0], axis=0)
# XT = torch.tensor(XTS)
# XT = XT.to(device)  

# uOut = model(XT) 
# wsOut = WatershedCut(uOut)
# squOut = torch.squeeze(wsOut)   
# xOut = squOut[0]
# yOut = squOut[1]
    
# npXPred = xOut.cpu().detach().numpy()
# npYPred = yOut.cpu().detach().numpy()

# img = GetConnectedComponentImage(npXPred, npYPred, 0.0)
# plt.figure(6)
# plt.imshow(img)
# plt.show()

# exit()




# Saving a Model
#torch.save(model.state_dict(), MODEL_PATH)
# Loading the model.
#checkpoint = torch.load(MODEL_PATH)
#model.load_state_dict(checkpoint)



# def GetRandWeights(xPred, yPred, nodeLabels):
#     W = nodeLabels.shape[0]
#     H = nodeLabels.shape[1]

#     G = nx.grid_2d_graph(W, H) 
#     nlabels_dict = dict()
#     for u, v, d in G.edges(data = True):
#         if u[0] == v[0]:    #  vertical edges
#             d['weight'] =  yPred[u[0], u[1]]
#         else:               # horizontal edges
#             d['weight'] =  xPred[u[0], u[1]]
                
#         nlabels_dict[u] = nodeLabels[u[0], u[1]]
#         nlabels_dict[v] = nodeLabels[v[0], v[1]]

#     # Just use fixed threshold for now    
#     bestT = 0.0
#     [posCounts, negCounts, mstEdges, totalPos, totalNeg] = ev.FindRandCounts(G, nlabels_dict)
    
#     # Inferning version for later
#     #[bestT, lowE, posCounts, negCounts, mstEdges, mstEdgeWeights, totalPos, totalNeg] = ev.FindMinEnergyAndRandCounts(G, nlabels_dict)
        
#     # Now we assign MST edge errors back to UNET  
#     xCounts = np.zeros((2, W, H), np.single)
#     yCounts = np.zeros((2, W, H), np.single)
        
#     for i in range(len(posCounts)):
#         (u,v) = mstEdges[i]
#         edgeDiff = posCounts[i] - negCounts[i]
#         edgeCount = posCounts[i] + negCounts[i]
#         if edgeDiff > 0.0:      
#             edgeSign = 1.0
#             edgeWeight = edgeDiff / edgeCount
#         else: 
#             edgeSign = -1.0
#             edgeWeight = -edgeDiff / edgeCount

#         if u[0] == v[0]:    #  vertical edges
#             yCounts[0, u[0], u[1]] = edgeSign                
#             yCounts[1, u[0], u[1]] = edgeWeight
#         else:               # horizontal edges
#             xCounts[0, u[0], u[1]] = edgeSign
#             xCounts[1, u[0], u[1]] = edgeWeight
                
#     return [xCounts, yCounts, bestT, totalPos, totalNeg]

# def GetConnectedComponentImage(xPred, yPred, threshold):

#     G = nx.grid_2d_graph(xPred.shape[0], xPred.shape[1])
#     for u, v, d in G.edges(data = True):
#         if u[0] == v[0]:    #  vertical edges
#             d['weight'] =  yPred[u[0], u[1]]
#         else:               # horizontal edges
#             d['weight'] =  xPred[u[0], u[1]]

#     L = ev.GetLabelsAtThreshold(G, threshold)
#     img = np.zeros((xPred.shape[0], xPred.shape[1]), np.single)
#     nlabel = dict()
#     for i in range(img.shape[0]):        
#         for j in range(img.shape[1]):
#             img[i,j] = L[(i,j)]
#             nlabel[L[(i,j)] ] = 1

#     print('Unique labels: ' + str(len(nlabel)))
#      return(img)

# def RandLossV2(uOut, yones, nodeLabels):

#     # We have two outputs from the unet 
#     # One corresponds to xedges
#     # One corresponds to yedges
#     # Code is particularly ugly since it handles them separately... 

#     squOut = torch.squeeze(uOut)   
#     xOut = squOut[0]
#     yOut = squOut[1]
    
#     npXPred = xOut.cpu().detach().numpy()
#     npYPred = yOut.cpu().detach().numpy()
                    
#     [xCounts, yCounts, bestT, totalPos, totalNeg] = GetRandWeights(npXPred, npYPred, nodeLabels)
#     #print('Total +ve weight: %f   -ve weight: %f' % (totalPos, totalNeg))
#     xLabels = torch.tensor(xCounts[0])
#     xWeights = torch.tensor(xCounts[1])
#     yLabels = torch.tensor(yCounts[0])
#     yWeights = torch.tensor(yCounts[1])
    
#     # Squared loss just because it was the easiest thing for me to code without looking stuff up! 
#     xErrors = (xOut.cpu() - xLabels) ** 2
#     xCon = torch.mul(xErrors, xWeights)
#     xloss = torch.sum(xCon)
    
#     yErrors = (yOut.cpu() - yLabels) ** 2
#     yCon = torch.mul(yErrors, yWeights)
#     yloss = torch.sum(yCon)
    
#     randLoss = xloss + yloss
#     return(randLoss)    
