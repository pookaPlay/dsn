import torch
import torch.nn.functional as F
from unet import UNet
import QuanSynData
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())


def RandLossDSN(uOut, yones, nodeLabels):

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

def EvalDSN(G):
    WG = G.copy()    
    CG = nx.Graph()

    wsLabel = dict()
    wsCount = dict()
    ccLabel = dict()
    ccMax = dict()

    mstWeight = list()
    mstEdge = list()

    labelUpto = 1

    for n in WG:        
        wsCount[n] = 0
        wsLabel[n] = 0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=False, key=lambda edge: edge[2]) 
    print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )

    upto = 0
    for u, v, w in sortedEdges:

        # new basin
        if (wsCount[u] == 0) and (wsCount[v] == 0): 
            wsCount[u] = 1
            wsCount[v] = 1
            wsLabel[u] = labelUpto
            wsLabel[v] = labelUpto
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            CG.add_node(labelUpto, weight = w)
            labelUpto = labelUpto + 1
        elif (wsCount[u] == 0):
            wsCount[u] = 1
            wsCount[v] = wsCount[v] + 1
            wsLabel[u] = wsLabel[v]
            WG.nodes[u]['label'] = WG.nodes[v]['label']
        elif (wsCount[v] == 0):
            wsCount[v] = 1
            wsCount[u] = wsCount[u] + 1
            wsLabel[v] = wsLabel[u]
            WG.nodes[v]['label'] = WG.nodes[u]['label']
        else:   
            nu = wsLabel[u]
            nv = wsLabel[v]

            if (nu != nv):
                depth = w - min(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
                CG.add_edge(wsLabel[u], wsLabel[v], weight = depth)

    print("Watershed has " + str(labelUpto-1) + " basins")

    ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    ccSorted = sorted(ccWeights, reverse=False, key=lambda edge: edge[2]) 
    print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

    # apply predefined threshold
    thresholdi = int(len(ccWeights)*0.75)
    threshold = ccSorted[thresholdi][2]    
    ccThresh = [ [d[0], d[1], threshold - d[2]] for d in ccSorted]
    #print("CCThresh is " + str(ccThresh[0]) + " -> " + str(ccThresh[-1]) )

    # Now run correlation clustering to find threshold
    print("Correlation Clustering at threshold " + str(threshold))
    mySets = nx.utils.UnionFind()   
    nextNode = dict()
    for n in CG:
        nextNode[n] = mySets[n]
    
    totalPos = sum([d[2] for d in ccThresh if d[2] > 0])
    totalNeg = sum([d[2] for d in ccThresh if d[2] < 0])
    accTotal = [0]*len(ccThresh)

    accTotal[0] = totalPos + totalNeg
    #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))
    DELTA_TOLERANCE = 1.0e-6
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = ccThresh[0][2] + DELTA_TOLERANCE

    for u, v, w in ccThresh:
        # Only need to go to zero weight
        if w <= 0.0:
            break
        if mySets[u] != mySets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in CG[cu]:                
                    if mySets[uev] == mySets[v]:
                        threshWeight = threshold - CG[cu][uev]['weight']
                        accWeight = accWeight + threshWeight
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            mySets.union(u, v)
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = w - DELTA_TOLERANCE


            ei = ei + 1
        
    print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowT))     

    # threshold graph and run connected components 
    LG = CG.copy()    
    #LG.remove_edges_from([(u,v) for (u,v,d) in  CG.edges(data=True) if (threshold - d['weight']) < lowT])
    LG.remove_edges_from([(u,v) for (u,v,d) in  ccThresh if d < lowT])
    L = {node:color for color,comp in enumerate(nx.connected_components(LG)) for node in comp}
    
    seenLabel = dict()
    count = 0
    for n in L:        
        CG.nodes[n]['label'] = L[n]
        if L[n] not in seenLabel:
            count = count + 1
            seenLabel[L[n]] = 1
    print("Final Segmentation has " + str(count) + " labels")

    return(WG, CG)


def ApplyDSN(G):
    WG = G.copy()    
    CG = nx.Graph()

    wsLabel = dict()
    wsCount = dict()
    ccLabel = dict()
    ccMax = dict()

    mstWeight = list()
    mstEdge = list()

    labelUpto = 1

    for n in WG:        
        wsCount[n] = 0
        wsLabel[n] = 0


    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=False, key=lambda edge: edge[2]) 
    print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )

    upto = 0
    for u, v, w in sortedEdges:

        # new basin
        if (wsCount[u] == 0) and (wsCount[v] == 0): 
            wsCount[u] = 1
            wsCount[v] = 1
            wsLabel[u] = labelUpto
            wsLabel[v] = labelUpto
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            CG.add_node(labelUpto, weight = w)
            labelUpto = labelUpto + 1
        elif (wsCount[u] == 0):
            wsCount[u] = 1
            wsCount[v] = wsCount[v] + 1
            wsLabel[u] = wsLabel[v]
            WG.nodes[u]['label'] = WG.nodes[v]['label']
        elif (wsCount[v] == 0):
            wsCount[v] = 1
            wsCount[u] = wsCount[u] + 1
            wsLabel[v] = wsLabel[u]
            WG.nodes[v]['label'] = WG.nodes[u]['label']
        else:   
            nu = wsLabel[u]
            nv = wsLabel[v]

            if (nu != nv):
                depth = w - min(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
                CG.add_edge(wsLabel[u], wsLabel[v], weight = depth)

    print("Watershed has " + str(labelUpto-1) + " basins")

    ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    ccSorted = sorted(ccWeights, reverse=False, key=lambda edge: edge[2]) 
    print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

    # apply predefined threshold
    thresholdi = int(len(ccWeights)*0.75)
    threshold = ccSorted[thresholdi][2]    
    ccThresh = [ [d[0], d[1], threshold - d[2]] for d in ccSorted]
    #print("CCThresh is " + str(ccThresh[0]) + " -> " + str(ccThresh[-1]) )

    # Now run correlation clustering to find threshold
    print("Correlation Clustering at threshold " + str(threshold))
    mySets = nx.utils.UnionFind()   
    nextNode = dict()
    for n in CG:
        nextNode[n] = mySets[n]
    
    totalPos = sum([d[2] for d in ccThresh if d[2] > 0])
    totalNeg = sum([d[2] for d in ccThresh if d[2] < 0])
    accTotal = [0]*len(ccThresh)

    accTotal[0] = totalPos + totalNeg
    #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))
    DELTA_TOLERANCE = 1.0e-6
    ei = 1      # edge index
    lowE = accTotal[0]
    lowT = ccThresh[0][2] + DELTA_TOLERANCE

    for u, v, w in ccThresh:
        # Only need to go to zero weight
        if w <= 0.0:
            break
        if mySets[u] != mySets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in CG[cu]:                
                    if mySets[uev] == mySets[v]:
                        threshWeight = threshold - CG[cu][uev]['weight']
                        accWeight = accWeight + threshWeight
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            mySets.union(u, v)
            # Swap next pointers... this incrementally builds a pointer cycle around all the nodes in the component
            tempNext = nextNode[u]
            nextNode[u] = nextNode[v]
            nextNode[v] = tempNext

            accTotal[ei] = accTotal[ei-1] - accWeight            
            #print("Energy at threshold " + str(w) + ": " + str(accTotal[ei]))
            if accTotal[ei] < lowE:
                lowE = accTotal[ei]
                lowT = w - DELTA_TOLERANCE


            ei = ei + 1
        
    print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowT))     

    # threshold graph and run connected components 
    LG = CG.copy()    
    #LG.remove_edges_from([(u,v) for (u,v,d) in  CG.edges(data=True) if (threshold - d['weight']) < lowT])
    LG.remove_edges_from([(u,v) for (u,v,d) in  ccThresh if d < lowT])
    L = {node:color for color,comp in enumerate(nx.connected_components(LG)) for node in comp}
    
    seenLabel = dict()
    count = 0
    for n in L:        
        CG.nodes[n]['label'] = L[n]
        if L[n] not in seenLabel:
            count = count + 1
            seenLabel[L[n]] = 1
    print("Final Segmentation has " + str(count) + " labels")

    return(WG, CG)


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

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')

if __name__ == '__main__':
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

    XS = np.expand_dims(Xsyn[0][0], axis=0)
    XS = np.expand_dims(XS, axis=0)
    #print(XS.shape)
    #XS = np.expand_dims(Xsyn[0], axis=0)
    YL = Ysyn[0]
    W = XS.shape[2]
    H = XS.shape[3]

    X = torch.tensor(XS, requires_grad=True)
    y = torch.ones([1, 2, W, H], dtype=torch.float32)

    X = X.to(device) 

    ## Here we run a fixed edge detector instead of a neural network
    [G_x, G_y, GXY] = Sobel(X)
    GXY1 = torch.squeeze(GXY)   
    npGXY = GXY1.cpu().detach().numpy()
    ScaleAndShow(npGXY, 4)

    #G_x1 = torch.squeeze(G_x)   
    #npG_x = G_x1.cpu().detach().numpy()
    #G_y1 = torch.squeeze(G_y)   
    #npG_y = G_y1.cpu().detach().numpy()
    #ScaleAndShow(npG_x, 5)
    #ScaleAndShow(npG_y, 6)
    
    print("####################################")
    print("ApplyDSN")
    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    for u, v, d in G.edges(data = True):
        d['weight'] =  (npGXY[u[0], u[1]] + npGXY[v[0], v[1]])/2.0
    
    # Run the DSN
    [WG, CG] = ApplyDSN(G)
    
    # Look at the segmentation
    wsImg, ccImg = GetSegImages(WG, CG, W, H)
    ScaleAndShow(wsImg, 7)
    ScaleAndShow(ccImg, 8)
    plt.show()

    print("DONE")
