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
model = UNet(in_channels=3, n_classes=1, depth=3, padding=True, up_mode='upsample').to(device)
optim = torch.optim.Adam(model.parameters())

def ApplyDSN(uOut):
    netOut = torch.squeeze(uOut).cpu()
    npGXY = netOut.detach().numpy()
    W = npGXY.shape[0]
    H = npGXY.shape[0]
    print("ApplyDSN to image  " + str(W) + ", " + str(H))

    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    for u, v, d in G.edges(data = True):
        d['weight'] =  (npGXY[u[0], u[1]] + npGXY[v[0], v[1]])/2.0

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

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')

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
    wsfirstEdge = dict()
    wsposCount = dict()
    wsnegCount = dict()
    ################################################################################################
    ## Watershed-Cuts in first layer
    labelUpto = 1
    for n in WG:        
        wsCount[n] = 0
        wsLabel[n] = 0        
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0
        wsposCount[n] = 0.0
        wsnegCount[n] = 0.0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=False, key=lambda edge: edge[2]) 
    print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
    
    for u, v, w in sortedEdges:
        su = wsSets[u]
        sv = wsSets[v]
        if su != sv:
            if (wsCount[u] == 0) or (wsCount[v] == 0): 
                labelAgreement = DotProductLabels( labelCount[su], labelCount[sv] )
                numLabelsU = GetNumberLabels( labelCount[su] )
                numLabelsV = GetNumberLabels( labelCount[sv] )
                labelDisagreement = numLabelsU * numLabelsV - labelAgreement
                
                allLabels = CombineLabels(labelCount[su], labelCount[sv])
                wsSets.union(u, v)                
                labelCount[ wsSets[u] ] = allLabels.copy()
                # Basin specific counts
                wsposCount[ wsSets[u] ] = wsposCount[ wsSets[u] ] + labelAgreement
                wsnegCount[ wsSets[u] ] = wsnegCount[ wsSets[u] ] + labelDisagreement

        # new basin
        if (wsCount[u] == 0) and (wsCount[v] == 0): 
            wsCount[u] = 1
            wsCount[v] = 1
            wsLabel[u] = labelUpto
            wsLabel[v] = labelUpto
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            wsfirstNode[labelUpto] = u           # Save one WS node to access the labelCounts from CG
            wsfirstEdge[labelUpto] = [u, v]
            CG.add_node(labelUpto, weight = w)  # One node in second graph for each WS basin
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
                CG.add_edge(nu, nv, weight = depth)
                CG.edges[nu, nv]['edge'] = [u, v]

    print("Watershed has " + str(labelUpto-1) + " basins")

    ################################################################################################
    ## Correlation clustering on Connected Components to find threshold
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
        
    print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowT))     

    ################################################################################################
    ## Final Connected Components at Correlation Clustering Threshold    
    ccSets = nx.utils.UnionFind()       
    cclabelCount = dict()
    wsEdge = dict()
    wsPos = dict()
    wsNeg = dict()
    
    totalPosWS = 0.0
    totalNegWS = 0.0
    for n in CG:
        # Setup the sets for CC
        wsIndex = wsSets[ wsfirstNode[n] ]                 
        cclabelCount[n] = labelCount[ wsIndex  ].copy()
        # Get counts labels for WS         
        wsEdge[n] = wsfirstEdge[n]
        wsPos[n] = wsposCount[wsIndex]
        wsNeg[n] = wsnegCount[wsIndex]
        totalPosWS = totalPosWS + wsPos[n]
        totalNegWS = totalNegWS + wsNeg[n]

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

    #LG.remove_edges_from([(u,v) for (u,v,d) in  ccThresh if d < lowT])
    posError = totalPos - threshPos
    negError = threshNeg
    randError = (posError + negError) / (totalPos + totalNeg)
    print("-----------------------------"); 
    print("Rand Error: " + str(randError))
    print("From #pos: " + str(totalPos) + " #neg: " + str(totalNeg))
    print("   and FN: " + str(posError) + "   FP: " + str(negError))

    finalThreshold = threshold - lowT        
    print("Final Threshold: "  + str(finalThreshold))
    print("-----------------------------"); 

    ######################################################
    ## Now Assign Errors back to image (neural net output)    
    print("Assigning Errors")
    labels = np.zeros((W, H), np.single)
    weights = np.zeros((W, H), np.single)        
    
    for n in wsEdge:
        [u, v] = wsEdge[n]
        #finalEdgeVal =  - WG.edges[cu, uev]['weight']
        if wsPos[n] >= wsNeg[n]:
            label = 1
            weight = (wsPos[n] - wsNeg[n])/ (wsPos[n] + wsNeg[n])
        else:
            label = -1
            weight = (wsNeg[n] - wsPos[n])/ (wsPos[n] + wsNeg[n])
        labels[u[0], u[1]] = label
        weights[u[0], u[1]] = weight
        labels[v[0], v[1]] = label
        weights[v[0], v[1]] = weight


    for n in range(len(ccEdge)):
        [u, v] = ccEdge[n]
        if ccPos[n] >= ccNeg[n]:
            label = 1
            weight = (ccPos[n] - ccNeg[n])/ (ccPos[n] + ccNeg[n])
        else:
            label = -1
            weight = (ccNeg[n] - ccPos[n])/ (ccPos[n] + ccNeg[n])
        labels[u[0], u[1]] = label
        weights[u[0], u[1]] = weight
        labels[v[0], v[1]] = label
        weights[v[0], v[1]] = weight

    return [finalThreshold, labels, weights]

def RandLossDSN(uOut, nodeLabels):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    netOut = torch.squeeze(uOut).cpu()
    npGXY = netOut.detach().numpy()
    #ScaleAndShow(npGXY, 4)
    
    # Setup input graph 
    G = nx.grid_2d_graph(W, H)
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        d['weight'] =  (npGXY[u[0], u[1]] + npGXY[v[0], v[1]])/2.0
        nlabels_dict[u] = nodeLabels[u[0], u[1]]
        nlabels_dict[v] = nodeLabels[v[0], v[1]]
    
    # Run the DSN
    [threshold, labels, weights] = EvalDSN(G, nlabels_dict, W, H)
    
    # Calc loss function 
    finalThreshold = torch.tensor(threshold)
    tlabels = torch.tensor(labels)
    tweights = torch.tensor(weights)
    
    finalOut = finalThreshold - netOut 
    
    # Hinge loss
    hinge_loss = 1 - torch.mul(finalOut, tlabels)
    hinge_loss[hinge_loss < 0] = 0
    werror = torch.mul(hinge_loss, tweights)
    loss = torch.sum(werror)      
    # Squared loss
    #error = (finalOut - tlabels) ** 2
    #werror = torch.mul(error, tweights)
    #loss = torch.sum(werror)      
    
    return(loss)    

##############################################################
## Basic Training Program
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

    # This expects 3 color input
    XS = np.expand_dims(Xsyn[0], axis=0)
    print(XS.shape)
    YL = Ysyn[0]
    W = XS.shape[2]
    H = XS.shape[3]

    X = torch.tensor(XS, requires_grad=True)    
    
    X = X.to(device) 
    
    epochs = 1000

    for epoch in range(epochs):

        uOut = model(X)  
        
        loss = RandLossDSN(uOut, YL)    

        myLoss = loss.item()
        print('Epoch %5d loss: %.3f' % (epoch+1, myLoss))

        optim.zero_grad()
        loss.backward()
        optim.step()
    

    print("DONE")
