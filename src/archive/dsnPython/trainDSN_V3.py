import torch
import torch.nn.functional as F
from unet import UNet
import QuanSynData
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, n_classes=2, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())
verbose = 1

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

############################################################################################################################
############################################################################################################################
############################################################################################################################
## DSN Inference Methods
############################################################################################################################
############################################################################################################################

def ApplyDSN(uOut):    
    netOut = torch.squeeze(uOut).cpu()
    if len(netOut.shape) == 2:
        netOut = netOut.unsqueeze(0)
        numInputs = 1
    else:
        numInputs = 2

    npGXY = netOut.detach().numpy()
    W = npGXY.shape[1]
    H = npGXY.shape[2]
    if verbose:
        print("ApplyDSN to image  " + str(W) + ", " + str(H))

    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    if numInputs == 1:
        for u, v, d in G.edges(data = True):
            d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0
    else:
        for u, v, d in G.edges(data = True):
            if u[0] == v[0]:    #  vertical edges
                d['weight'] =  npGXY[1, u[0], u[1]]
            else:               # horizontal edges
                d['weight'] =  npGXY[0, u[0], u[1]]

    [WG, CG] = ApplyDSNGraph(G)
    wsImg, ccImg = GetSegImages(WG, CG, W, H)
    return (wsImg, ccImg)

def ApplyDSNGraph(G):
    WG = G.copy()    
    CG = nx.Graph()

    wsLabel = dict()
    wsCount = dict()


    labelUpto = 1
    for n in WG:        
        wsCount[n] = 0
        wsLabel[n] = 0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=False, key=lambda edge: edge[2]) 
    if verbose:
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

    if verbose:
        print("Watershed has " + str(labelUpto-1) + " basins")

    ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    ccSorted = sorted(ccWeights, reverse=False, key=lambda edge: edge[2]) 
    if verbose:
        print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

    # apply predefined threshold
    thresholdi = int(len(ccWeights)*0.75)
    threshold = ccSorted[thresholdi][2]    
    ccThresh = [ [d[0], d[1], threshold - d[2]] for d in ccSorted]
    #print("CCThresh is " + str(ccThresh[0]) + " -> " + str(ccThresh[-1]) )

    # Now run correlation clustering to find threshold
    if verbose:
        print("Correlation Clustering at threshold " + str(threshold))
    threshSets = nx.utils.UnionFind()   
    nextNode = dict()
    for n in CG:
        nextNode[n] = threshSets[n]
    
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
        if threshSets[u] != threshSets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in CG[cu]:                
                    if threshSets[uev] == threshSets[v]:
                        threshWeight = threshold - CG[cu][uev]['weight']
                        accWeight = accWeight + threshWeight
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            threshSets.union(u, v)
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
    
    if verbose:        
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
    if verbose:
        print("Final Segmentation has " + str(count) + " labels")

    return(WG, CG)


############################################################################################################################
############################################################################################################################
############################################################################################################################
## Kruskal DSN Eval with disjoint sets datastructure
############################################################################################################################
############################################################################################################################

def DotProductLabels(a, b):
    ssum = 0.0    
    for key in a: 
        if key in b: 
            ssum = ssum + a[key]*b[key]
            
    return ssum

def GetNumberLabels(a):
    ssum = 0.0    
    for key in a:         
        ssum = ssum + a[key]
            
    return ssum

def CombineLabels(a, b):
    c = a.copy()
    
    for key in b:         
        if key in c:
            c[key] = c[key] + b[key]
        else:
            c[key] = b[key]
            
    return c

def EvalDSN(G, nlabels_dict, W, H):
    WG = G.copy()    
    CG = nx.Graph()
    wsSets = nx.utils.UnionFind()       

    labelCount = dict()    
    wsLabel = dict()
    wsCount = dict()
    wsfirstNode = dict()
    wsEdge = dict()
    wsPos = dict()
    wsNeg = dict()
    
    if verbose:
        print("-----------------------------"); 
    ################################################################################################
    ## Watershed-Cuts in first layer
    labelUpto = 1
    for n in WG:        
        wsCount[n] = 0
        wsLabel[n] = 0        
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=False, key=lambda edge: edge[2]) 
    if verbose:
        print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
    
    for u, v, w in sortedEdges:
        lu = -1
        if (wsCount[u] == 0) and (wsCount[v] == 0):     # new basin
            wsCount[u] = 1
            wsCount[v] = 1
            wsLabel[u] = labelUpto
            wsLabel[v] = labelUpto
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            wsEdge[labelUpto] = list()
            wsPos[labelUpto] = list()
            wsNeg[labelUpto] = list()
            wsfirstNode[labelUpto] = u           # Save one WS node to access the labelCounts from CG            
            CG.add_node(labelUpto, weight = w)  # One node in second graph for each WS basin
            lu = labelUpto
            labelUpto = labelUpto + 1
        elif (wsCount[u] == 0):                       # extend basin
            wsCount[u] = 1
            wsCount[v] = wsCount[v] + 1
            wsLabel[u] = wsLabel[v]                    
            WG.nodes[u]['label'] = WG.nodes[v]['label']
            lu = wsLabel[v]
        elif (wsCount[v] == 0):                       # extend basin
            wsCount[v] = 1
            wsCount[u] = wsCount[u] + 1
            wsLabel[v] = wsLabel[u]
            WG.nodes[v]['label'] = WG.nodes[u]['label']                    
            lu = wsLabel[u]
        else:   
            nu = wsLabel[u]
            nv = wsLabel[v]

            if (nu != nv):
                depth = w - min(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
                CG.add_edge(nu, nv, weight = depth)
                CG.edges[nu, nv]['edge'] = [u, v]

        su = wsSets[u]
        sv = wsSets[v]
        if su != sv:
            if lu > 0:        
                labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
                numLabelsU = GetNumberLabels( labelCount[su] )
                numLabelsV = GetNumberLabels( labelCount[sv] )
                labelDisagreement = numLabelsU * numLabelsV - labelAgreement
                
                allLabels = CombineLabels(labelCount[su], labelCount[sv])
                wsSets.union(u, v)                
                labelCount[ wsSets[u] ] = allLabels.copy()
                
                wsEdge[lu].append([u,v])
                wsPos[lu].append(labelAgreement)
                wsNeg[lu].append(labelDisagreement)                


    if verbose:
        print("Watershed has " + str(labelUpto-1) + " basins")

    ################################################################################################
    ## Correlation clustering on Connected Components to find threshold
    ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    ccSorted = sorted(ccWeights, reverse=False, key=lambda edge: edge[2]) 
    if verbose:
        print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

    # apply predefined threshold
    thresholdi = int(len(ccWeights)*0.75)
    threshold = ccSorted[thresholdi][2]    
    ccThresh = [ [d[0], d[1], threshold - d[2]] for d in ccSorted]
    #print("CCThresh is " + str(ccThresh[0]) + " -> " + str(ccThresh[-1]) )

    # Now run correlation clustering to find threshold
    if verbose:
        print("Correlation Clustering at threshold " + str(threshold))

    threshSets = nx.utils.UnionFind()   
    nextNode = dict()
    for n in CG:
        nextNode[n] = threshSets[n]
    
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
        if threshSets[u] != threshSets[v]:
            accWeight = 0.0
            # traverse nodes in u and look at edges
            # if fully connected we should probably traverse nodes u and v instead
            done = False
            cu = u
            while not done:
                for uev in CG[cu]:                
                    if threshSets[uev] == threshSets[v]:
                        threshWeight = threshold - CG.edges[cu, uev]['weight']
                        accWeight = accWeight + threshWeight
                cu = nextNode[cu]
                if cu == u:
                    done = True

            # Merge sets
            threshSets.union(u, v)
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

    if verbose:        
        print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowT))     

    ################################################################################################
    ## Final Connected Components at Correlation Clustering Threshold    
    ccSets = nx.utils.UnionFind()       
    cclabelCount = dict()
    basinPos = dict()
    basinNeg = dict()
    totalPosWS = 0.0
    totalNegWS = 0.0
    for n in CG:
        # Setup the sets for CC
        wsIndex = wsSets[ wsfirstNode[n] ]                 
        cclabelCount[n] = labelCount[ wsIndex  ].copy()
        # Accumulate counts for each basin
        #print(wsPos[n])
        basinPos[n] = sum([d for d in wsPos[n]])
        basinNeg[n] = sum([d for d in wsNeg[n]])
        #print("Basin  " + str(n) + " Pos: " + str(basinPos[n]) + "   and Neg: " + str(basinNeg[n]))
        totalPosWS = totalPosWS + basinPos[n]
        totalNegWS = totalNegWS + basinNeg[n]

    ccEdge = list()    
    ccBasin = list()    
    ccPos = list()
    ccNeg = list()
    totalPos = totalPosWS
    totalNeg = totalNegWS
    threshPos = totalPosWS
    threshNeg = totalNegWS

    for u, v, w in ccThresh:
        su = ccSets[u]
        sv = ccSets[v]
        if su != sv:
            labelAgreement = DotProductLabels( cclabelCount[su], cclabelCount[sv] )
            numLabelsU = GetNumberLabels( cclabelCount[su] )
            numLabelsV = GetNumberLabels( cclabelCount[sv] )
            labelDisagreement = numLabelsU * numLabelsV - labelAgreement                
            allLabels = CombineLabels(cclabelCount[su], cclabelCount[sv])
            ccSets.union(u, v)                
            cclabelCount[ ccSets[u] ] = allLabels.copy()
            # Basin specific counts
            ccBasin.append([u,v])
            ccEdge.append(CG[u][v]['edge'])            
            ccPos.append(labelAgreement)
            ccNeg.append(labelDisagreement)
            totalPos = totalPos + labelAgreement
            totalNeg = totalNeg + labelDisagreement
            if w > lowT:
                threshPos = threshPos + labelAgreement
                threshNeg = threshNeg + labelDisagreement
    
    posError = totalPos - threshPos
    negError = threshNeg
    randError = (posError + negError) / (totalPos + totalNeg)
    if verbose:        
        print("Rand Error: " + str(randError))
        print("From #pos: " + str(totalPos) + " #neg: " + str(totalNeg))
        print("   and FN: " + str(posError) + "   FP: " + str(negError))

    finalThreshold = threshold - lowT        

    if verbose:
        print("Final Threshold: "  + str(finalThreshold))
        print("-----------------------------"); 

    ######################################################
    ## Now Assign Errors back to image (neural net output)    
    if verbose:
        print("Assigning Errors")

    labels = np.zeros((2, W, H), np.single)
    weights = np.zeros((2, W, H), np.single)        
    
    for n in wsEdge:
        for i in range(len(wsEdge[n])):
            [u, v] = wsEdge[n][i]
            
            if wsPos[n][i] >= wsNeg[n][i]:
                label = 1
                weight = (wsPos[n][i] - wsNeg[n][i])/ (wsPos[n][i] + wsNeg[n][i])         
            else:
                label = -1
                weight = (wsNeg[n][i] - wsPos[n][i])/ (wsPos[n][i] + wsNeg[n][i])        

            if u[0] == v[0]:    #  vertical edges
                labels[1, u[0], u[1]] = label
                weights[1, u[0], u[1]] = weight
            else:               # horizontal edges
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight



    for n in range(len(ccEdge)):
        [u, v] = ccEdge[n]
        if ccPos[n] >= ccNeg[n]:
            label = 1
            weight = (ccPos[n] - ccNeg[n])/ (ccPos[n] + ccNeg[n])            
        else:
            label = -1
            weight = (ccNeg[n] - ccPos[n])/ (ccPos[n] + ccNeg[n])
            
        if u[0] == v[0]:    #  vertical edges
            labels[1, u[0], u[1]] = label
            weights[1, u[0], u[1]] = weight
        else:               # horizontal edges
            labels[0, u[0], u[1]] = label
            weights[0, u[0], u[1]] = weight

    return [finalThreshold, labels, weights, randError]

def RandLossDSN(uOut, nodeLabels, epoch):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    netOut = torch.squeeze(uOut).cpu()
    if len(netOut.shape) == 2:
        netOut = netOut.unsqueeze(0)
        numInputs = 1
    else:
        numInputs = 2
    npGXY = netOut.detach().numpy()    
    
    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    nlabels_dict = dict()
    if numInputs == 1:
        for u, v, d in G.edges(data = True):
            d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0
            nlabels_dict[u] = nodeLabels[u[0], u[1]]
            nlabels_dict[v] = nodeLabels[v[0], v[1]]
    else:
        for u, v, d in G.edges(data = True):
            if u[0] == v[0]:    #  vertical edges
                d['weight'] =  npGXY[1, u[0], u[1]]
            else:               # horizontal edges
                d['weight'] =  npGXY[0, u[0], u[1]]
            nlabels_dict[u] = nodeLabels[u[0], u[1]]
            nlabels_dict[v] = nodeLabels[v[0], v[1]]
            
    # Run the DSN
    [threshold, labels, weights, randError] = EvalDSN(G, nlabels_dict, W, H)

    # Apply final threshold 
    finalThreshold = torch.tensor(threshold)
    finalOut = finalThreshold - netOut 

    # Calc loss function     
    tlabels = torch.tensor(labels)
    tweights = torch.tensor(weights)    
    
    # Squared loss    
    errors = (finalOut - tlabels) ** 2
    werrors = torch.mul(errors, tweights)
    randLoss = torch.sum(werrors)
    
    # Hinge loss
    #hinge_loss = 1 - torch.mul(finalOut, tlabels)
    #hinge_loss[hinge_loss < 0] = 0
    #werror = torch.mul(hinge_loss, tweights)
    #loss = torch.sum(werror)      

    print("Epoch " + str(epoch) + ":   Loss " + str(randLoss.item()) + "  and Rand " + str(randError))
    return(randLoss)    


##############################################################
## Basic Training Program
if __name__ == '__main__':
    Xsyn, Ysyn, XTsyn, YTsyn = QuanSynData.GetData(1)
    print("Yah")
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

    ### This expects 3 color input
    XS3 = np.expand_dims(Xsyn[0], axis=0)

    ### These two lines convert 3 band input to 1 band
    XS1 = np.expand_dims(Xsyn[0][0], axis=0)
    XS1 = np.expand_dims(XS1, axis=0)

    YL = Ysyn[0]
    W = XS1.shape[2]
    H = XS1.shape[3]

    X1 = torch.tensor(XS1, requires_grad=True)    
    X3 = torch.tensor(XS3, requires_grad=True)    
    
    X1 = X1.to(device) 
    
    # Run Sobel for reference
    verbose = 1
    if verbose:
        print("####################################")
        print("Sobel Test")

    [G_x, G_y, GXY] = Sobel(X1)
    ### 2 output version
    uOut = torch.stack((torch.squeeze(G_x), torch.squeeze(G_y)))
    ### 1 output version
    #uOut = GXY
        
    print("DSN Train Loss")
    print("==============")
    loss = RandLossDSN(uOut, YL, 0)
        
    print("Rand Error Check")
    print("================")        
    wsImg, ccImg = ApplyDSN(uOut)
    error = ExhaustiveRand(ccImg, YL)
    
    ScaleAndShow(wsImg, 3)
    ScaleAndShow(ccImg, 4)    
    plt.show()    

    verbose = 0
    X3 = X3.to(device)                 
    epochs = 50

    for epoch in range(epochs):

        uOut = model(X3)  
        
        loss = RandLossDSN(uOut, YL, epoch)    

        optim.zero_grad()
        loss.backward()
        optim.step()
    

    print("DONE")
