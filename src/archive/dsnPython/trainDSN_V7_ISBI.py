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
import skimage.io as skio



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

    G_x = -torch.pow(G_x,2) 
    G_y = -torch.pow(G_y,2) 
    GXY = -torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
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
        print("ApplyDSN to image  " + str(W) + ", " + str(H) + " with " + str(numInputs) + " inputs")

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
    
    for n in WG:                        
        WG.nodes[n]['label'] = 0

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = False : small -> big
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    if verbose:
        print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )

    labelUpto = 1
    for u, v, w in sortedEdges:

        # new basin
        if (WG.nodes[u]['label'] == 0) and (WG.nodes[v]['label'] == 0): 
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            CG.add_node(labelUpto, weight = w)
            labelUpto = labelUpto + 1
        elif (WG.nodes[u]['label'] == 0):
            WG.nodes[u]['label'] = WG.nodes[v]['label']
        elif (WG.nodes[v]['label'] == 0):
            WG.nodes[v]['label'] = WG.nodes[u]['label']
        else:   
            nu = WG.nodes[u]['label']
            nv = WG.nodes[v]['label']

            if (nu != nv):
                if (CG.has_edge(nu, nv) == False):            
                    # Standard smallest depth is w - min(b1, b2)
                    # We want to merge smallest depth so we take the negative to make it big as good
                    depth = w - max(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
                    CG.add_edge(nu, nv, weight = depth)
    numBasins = labelUpto-1
    if verbose:
        print("Watershed has " + str(numBasins) + " basins")
    
    if (numBasins > 1):
        ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
        # reverse = False : small -> big
        ccSorted = sorted(ccWeights, reverse=True, key=lambda edge: edge[2]) 
        if verbose:
            print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

        # apply predefined threshold
        thresholdi = int(len(ccSorted) * THRESH_OFFSET)
        threshold = ccSorted[thresholdi][2]    
        ccThresh = [ [d[0], d[1], d[2] - threshold] for d in ccSorted]
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
        if verbose:
            print("Correlation Clustering totals +ve: " + str(totalPos) + ", -ve: " + str(totalNeg))
    
        accTotal[0] = totalPos + totalNeg
        #print("Energy 0: " + str(accTotal[0]) + " from Pos: " + str(totalPos) + ", Neg: " + str(totalNeg))
        DELTA_TOLERANCE = 1.0e-6
        ei = 1      # edge index
        lowE = accTotal[0]
        lowT = ccThresh[0][2] + 1.0e3
        prevT = lowT
        for u, v, w in ccThresh:
            # Only need to go to zero weight
            #if w >= 0.0:
            #    break
            if threshSets[u] != threshSets[v]:
                accWeight = 0.0
                # traverse nodes in u and look at edges
                # if fully connected we should probably traverse nodes u and v instead
                done = False
                cu = u
                while not done:
                    for uev in CG[cu]:                
                        if threshSets[uev] == threshSets[v]:
                            threshWeight = CG[cu][uev]['weight'] - threshold
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
                    lowT = (w + prevT) / 2.0

                prevT = w
                ei = ei + 1
        
        if verbose:        
            print("Smallest Energy: " + str(lowE) + " at threshold " + str(lowT))     
        
        # threshold graph and run connected components 
        finalThreshold = threshold + lowT
        if verbose:
            print("Final Threshold is: " + str(finalThreshold))

        LG = CG.copy()    
        LG.remove_edges_from([(u,v) for (u,v,d) in  LG.edges(data=True) if (d['weight'] - finalThreshold) < 0.0])
        #LG.remove_edges_from([(u,v) for (u,v,d) in  ccThresh if d < lowT])
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

    else:
        if verbose:        
            print("One basin at Energy: " + str(lowE) + " at threshold " + str(lowT) + " and label 1")     

        threshold = 0.0
        lowE = 0.0
        lowT = 0.0
                        
        for n in CG:        
            CG.nodes[n]['label'] = 1

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

def EvalDSN(G, nlabels_dict, W, H, numInputs):
    WG = G.copy()    
    CG = nx.Graph()
    wsSets = nx.utils.UnionFind()       

    labelCount = dict()            
    wsfirstNode = dict()
    wsEdge = dict()
    wsPos = dict()
    wsNeg = dict()
    
    if verbose:
        print("-----------------------------"); 
    
    ################################################################################################
    ## Watershed-Cuts in first layer        
    for n in WG:        
        WG.nodes[n]['label'] = 0
        labelCount[n] = dict()
        labelCount[n][ nlabels_dict[n] ] = 1.0        

    edgeWeights = [(u,v,w) for (u,v,w) in WG.edges(data = 'weight')]    
    # reverse = True : +ve -> -ve
    sortedEdges = sorted(edgeWeights, reverse=True, key=lambda edge: edge[2]) 
    if verbose:
        print("WS affinities: " + str(sortedEdges[0][2]) + " -> " + str(sortedEdges[-1][2]) )
    
    labelUpto = 1
    for u, v, w in sortedEdges:
        lu = -1
        if (WG.nodes[u]['label'] == 0) and (WG.nodes[v]['label'] == 0):     # new basin
            WG.nodes[u]['label'] = labelUpto
            WG.nodes[v]['label'] = labelUpto
            wsEdge[labelUpto] = list()
            wsPos[labelUpto] = list()
            wsNeg[labelUpto] = list()
            wsfirstNode[labelUpto] = u           # Save one WS node to access the labelCounts from CG            
            CG.add_node(labelUpto, weight = w)  # One node in second graph for each WS basin
            lu = labelUpto
            labelUpto = labelUpto + 1
        elif (WG.nodes[u]['label'] == 0):                       # extend basin            
            WG.nodes[u]['label'] = WG.nodes[v]['label']
            lu = WG.nodes[v]['label']
        elif (WG.nodes[v]['label'] == 0):                       # extend basin
            WG.nodes[v]['label'] = WG.nodes[u]['label']                    
            lu = WG.nodes[u]['label']
        else:   
            nu = WG.nodes[u]['label']
            nv = WG.nodes[v]['label']
            if (nu != nv):
                if (CG.has_edge(nu, nv) == False):       
                    # Standard smallest depth is w - min(b1, b2)
                    # We want to merge smallest depth so we take the negative to make it big as
                    depth = w - max(CG.nodes[nu]['weight'], CG.nodes[nv]['weight'])
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

    numBasins = labelUpto-1
    if verbose:
        print("Watershed has " + str(numBasins) + " basins")

    ##########################################
    ## Initialize basin stats and second layer
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

    if numBasins > 1:
        ################################################################################################
        ## Correlation clustering on Connected Components to find threshold
        ccWeights = [(u,v,w) for (u,v,w) in CG.edges(data = 'weight')]    
        # reverse = True : +ve -> -ve
        ccSorted = sorted(ccWeights, reverse=True, key=lambda edge: edge[2]) 
        if verbose:
            print("CC has " + str(len(ccWeights)) + " affinities: " + str(ccSorted[0][2]) + " -> " + str(ccSorted[-1][2]) )

        # apply predefined threshold
        thresholdi = int(len(ccSorted) * THRESH_OFFSET)
        threshold = ccSorted[thresholdi][2]    
        ccThresh = [ [d[0], d[1], d[2] - threshold] for d in ccSorted]
        #print("CCThresh is " + str(ccThresh[0]) + " -> " + str(ccThresh[-1]) )

        # Now run correlation clustering to find threshold
        if verbose:
            print("Correlation Clustering at threshold " + str(threshold))

        threshSets = nx.utils.UnionFind()   
        nextNode = dict()
        for n in CG:
            nextNode[n] = threshSets[n]
        
        totalPosCC = sum([d[2] for d in ccThresh if d[2] > 0])
        totalNegCC = sum([d[2] for d in ccThresh if d[2] < 0])
        accTotal = [0]*(len(ccThresh)+1)
        if verbose:
            print("Correlation Clustering totals +ve: " + str(totalPosCC) + ", -ve: " + str(totalNegCC))

        accTotal[0] = totalPosCC + totalNegCC
        
        DELTA_TOLERANCE = 1.0e-6
        ei = 1      # edge index
        lowE = accTotal[0]
        lowT = ccThresh[0][2] + 1.0e3
        prevT = lowT
        for u, v, w in ccThresh:
            # Only need to go to zero weight
            #if w <= 0.0:
            #    break            
            if threshSets[u] != threshSets[v]:
                accWeight = 0.0
                # traverse nodes in u and look at edges
                # if fully connected we should probably traverse nodes u and v instead
                done = False
                cu = u
                while not done:
                    for uev in CG[cu]:                
                        if threshSets[uev] == threshSets[v]:
                            threshWeight = CG.edges[cu, uev]['weight'] - threshold
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
                    lowT = (prevT + w)/2.0
                prevT = w
                ei = ei + 1
        if verbose:        
            print("Lowest Energy: " + str(lowE) + " at threshold " + str(lowT))     

        # threshold graph and run connected components 
        finalThreshold = threshold + lowT
        if verbose:
            print("Final Threshold is: " + str(finalThreshold))

        ################################################################################################
        ## Final Connected Components at Correlation Clustering Threshold                    
        for u, v, w in ccThresh:
            finalW = w - lowT
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
                if finalW >= 0.0:
                    threshPos = threshPos + labelAgreement
                    threshNeg = threshNeg + labelDisagreement        
    else:
        threshold = 0.0
        lowT = 0.0     
        finalThreshold = 0.0  
    
    posError = totalPos - threshPos
    negError = threshNeg
    randError = (posError + negError) / (totalPos + totalNeg)
    if verbose:        
        print("Rand Error: " + str(randError))
        print("From #pos: " + str(totalPos) + " #neg: " + str(totalNeg))
        print("   and FN: " + str(posError) + "   FP: " + str(negError))


    ######################################################
    ## Now Assign Errors back to image (neural net output)    
    if verbose:
        print("Assigning Errors")

    labels = np.zeros((numInputs, W, H), np.single)
    weights = np.zeros((numInputs, W, H), np.single)        
    
    for n in wsEdge:
        for i in range(len(wsEdge[n])):
            [u, v] = wsEdge[n][i]
            
            if wsPos[n][i] >= wsNeg[n][i]:
                label = 1
                weight = (wsPos[n][i] - wsNeg[n][i])/ (wsPos[n][i] + wsNeg[n][i]) 
            else:
                label = -1
                weight = (wsNeg[n][i] - wsPos[n][i])/ (wsPos[n][i] + wsNeg[n][i])        
            if numInputs == 1:
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randError
                labels[0, v[0], v[1]] = label
                weights[0, v[0], v[1]] = weight * randError
            else:
                if u[0] == v[0]:    #  vertical edges
                    labels[1, u[0], u[1]] = label
                    weights[1, u[0], u[1]] = weight * randError
                else:               # horizontal edges
                    labels[0, u[0], u[1]] = label
                    weights[0, u[0], u[1]] = weight * randError



    for n in range(len(ccEdge)):
        [u, v] = ccEdge[n]
        if ccPos[n] >= ccNeg[n]:
            label = 1
            weight = (ccPos[n] - ccNeg[n])/ (ccPos[n] + ccNeg[n])            
        else:
            label = -1
            weight = (ccNeg[n] - ccPos[n])/ (ccPos[n] + ccNeg[n])
        if numInputs == 1:
            labels[0, u[0], u[1]] = label
            weights[0, u[0], u[1]] = weight * randError
            labels[0, v[0], v[1]] = label
            weights[0, v[0], v[1]] = weight * randError
        else:            
            if u[0] == v[0]:    #  vertical edges
                labels[1, u[0], u[1]] = label
                weights[1, u[0], u[1]] = weight * randError
            else:               # horizontal edges
                labels[0, u[0], u[1]] = label
                weights[0, u[0], u[1]] = weight * randError

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
    
    if verbose:
        print("EvalDSN with image  " + str(W) + ", " + str(H) + " with " + str(numInputs) + " inputs")

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
    [threshold, labels, weights, randError] = EvalDSN(G, nlabels_dict, W, H, numInputs)

    # Apply final threshold 
    finalThreshold = torch.tensor(threshold)
    finalOut = netOut - finalThreshold

    # Calc loss function     
    tlabels = torch.tensor(labels)
    tweights = torch.tensor(weights)    
    
    # Squared loss    
    errors = (finalOut - tlabels) ** 2
    werrors = torch.mul(errors, tweights)
    randLoss = torch.sum(werrors)
    
    # Hinge loss
    #hinge_loss = 1.0 - torch.mul(finalOut, tlabels)
    #hinge_loss[hinge_loss < 0.0] = 0.0
    #werror = torch.mul(hinge_loss, tweights)
    #randLoss = torch.sum(werror)      

    print("Epoch " + str(epoch) + ":   Loss " + str(randLoss.item()) + "  and Rand " + str(randError))
    return(randLoss, randError)    


def AnalyzeGroundTruth(nodeLabels):
    W = nodeLabels.shape[0]
    H = nodeLabels.shape[1]

    # Setup input graph
    sameCount = 0.0
    diffCount = 0.0
    G = nx.grid_2d_graph(W, H)
    
    for u, v, d in G.edges(data = True):
        if nodeLabels[u[0], u[1]] == nodeLabels[v[0], v[1]]:
            sameCount = sameCount + 1
        else:
            diffCount = diffCount + 1
    fracSame = sameCount / (sameCount + diffCount)    
    return(fracSame, sameCount, diffCount)            

##############################################################
## Basic Training Program
if __name__ == '__main__':

    verbose = 0
    testRand = 0
    bsdsData = 1
    numEpochs = 5000
    THRESH_OFFSET = 0.75

    trn_img_path = "/home/vj/Dropbox/LosAlamos/dsn/data/ISBIChallenge/train-volume/0.tif"
    trn_seg_path = "/home/vj/Dropbox/LosAlamos/dsn/data/ISBIChallenge/train-labels/0.tif"

    tst_img_path = "/home/vj/Dropbox/LosAlamos/dsn/data/ISBIChallenge/train-volume/1.tif"
    tst_seg_path = "/home/vj/Dropbox/LosAlamos/dsn/data/ISBIChallenge/train-labels/1.tif"
    
    # Training image
    img1      = skio.imread(trn_img_path)
    img1      = img1[0:64, 0:64]
    seg1      = skio.imread(trn_seg_path)
    seg1      = 1*(seg1[0:64, 0:64] > 0)
    nclasses  = len(np.unique(seg1)) # it is 2

    # Testing image
    img2      = skio.imread(tst_img_path)
    img2      = img2[0:64, 0:64]
    seg2      = 1*(skio.imread(tst_seg_path) > 0)
    seg2      = seg2[0:64, 0:64]

    fig, axs = plt.subplots(2,2)
    axs[0][0].imshow(img1, cmap='gray')
    axs[0][0].set_title("Training image")

    axs[0][1].imshow(img2, cmap='gray')
    axs[0][1].set_title("Testing image")

    axs[1][0].imshow(seg1, cmap='gray')
    axs[1][0].set_title("Train labeled image")

    axs[1][1].imshow(seg2, cmap='gray')
    axs[1][1].set_title("Testing labeled image")
    


    Ysyn = np.expand_dims(seg1, axis=0)
    YTsyn = np.expand_dims(seg2, axis=0)
    Xsyn = np.zeros((1, img1.shape[0], img1.shape[1]))        
    XTsyn = np.zeros((1, img2.shape[0], img2.shape[1]))        
    Xsyn[0,:,:] = img1
    XTsyn[0,:,:] = img2
        
    Xsyn = np.expand_dims(Xsyn, axis=0)
    XTsyn = np.expand_dims(XTsyn, axis=0)

    Xsyn = Xsyn.astype(np.single)
    XTsyn = XTsyn.astype(np.single)
    Ysyn = Ysyn.astype(np.single)
    YTsyn = YTsyn.astype(np.single)


    # Setting up U-net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, n_classes=nclasses, depth=5, padding=True, up_mode='upsample').to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # was
    optimizer = torch.optim.Adam(model.parameters())
    verbose = 1
    THRESH_OFFSET = 0.75

    
    XS3 = np.expand_dims(Xsyn[0], axis=0)
    YL = Ysyn[0]    

    ### These two lines convert 3 band input to 1 band
    XS1 = np.expand_dims(Xsyn[0][0], axis=0)
    XS1 = np.expand_dims(XS1, axis=0)

    X1 = torch.tensor(XS1, requires_grad=False)    
    X3 = torch.tensor(XS3, requires_grad=False)    
    
    (fracSame, sameCount, diffCount) = AnalyzeGroundTruth(YL)
    print("Ground Truth Graph Ratio: " + str(fracSame) + "     #same/#diff: " + str(sameCount) + "/" + str(diffCount))

    if testRand:
        X1 = X1.to(device) 
        # Run Sobel for reference
        verbose = 1
        if verbose:
            print("####################################")
            print("Sobel Test")

        [G_x, G_y, GXY] = Sobel(X1)
        ### 2 output version
        #uOut = torch.stack((torch.squeeze(G_x), torch.squeeze(G_y)))
        ### 1 output version
        uOut = GXY
            
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
    
    else:    
        X3 = X3.to(device)                         
        
        bestError = 1    
        for epoch in range(numEpochs):

            uOut = model(X3)  
            
            loss, randError = RandLossDSN(uOut, YL, epoch)    
            
            if randError < bestError:
                bestError = randError            
                wsImg, ccImg = ApplyDSN(uOut)                             
                ScaleAndShow(ccImg, 4)            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
        
        print("Best error: " + str(bestError))
        plt.show()    

    print("DONE")
