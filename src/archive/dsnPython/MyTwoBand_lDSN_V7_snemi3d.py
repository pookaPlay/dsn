import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
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

# MATPLOTLIB defaults
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15



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

def RandLossDSN(uOut, nodeLabels, trn_idx):
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
    #finalOut = netOut - finalThreshold

    tlabels = torch.tensor(labels)
    tweights = torch.tensor(weights)    

    #ones = torch.ones((1, 128, 128))
    #print(ones.shape)
    #print(netOut.shape)
    #finalOut = torch.div(ones, netOut)
    #hinge_loss = torch.mul(finalOut, tlabels)
    #werror = torch.mul(hinge_loss, tweights)
    #minv = torch.min(netOut)
    #maxv = torch.max(netOut)
    #print("NetOut Range %f -> %f" % (minv, maxv))    
    #randLoss = torch.sum(werror)          
    # if trn_idx > 3:
    #     hinge_loss = 1.0 - torch.mul(netOut, tlabels)
    #     hinge_loss[hinge_loss < 0.0] = 0.0
    #     werror = torch.mul(hinge_loss, tweights)
    #     randLoss = torch.sum(werror)      
    # else:
    # Squared loss    
    errors = (netOut - tlabels) ** 2
    werrors = torch.mul(errors, tweights)
    randLoss = torch.sum(werrors)

    # Hinge loss
    #finalOut = torch.Tanh(netOut)

    #print("\t\t" + str(trn_idx) + " Loss " + str(randLoss.item()) + "  and Rand " + str(randError))
    return(randLoss, randError)    


def get_all_img_paths(pth, idxs):
    """
    Returns paths of all images having particular index
    """
    img_paths = []
    for idx in idxs:
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx)+'_0_0.tif')

    return img_paths



def get_train_img_paths(pth, num_stacks):
    """
    Returns a list of training image paths. The training images
    are assumed to be `stack_<idx>_0_0.tif
    """
    img_paths = []
    for idx in range(0,num_stacks):
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+10)+'_1_1.tif')
    return img_paths

def get_valid_img_paths(pth, num_stacks):
    """
    Returns a list of training image paths. The training images
    are assumed to be `stack_<idx>_0_1.tif
    """
    img_paths = []
    for idx in range(0,num_stacks):
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+10)+'_2_2.tif')

    return img_paths

def get_train_img_stack(pth, epth, num_stacks):
    """
    Returns a list of training image paths. The training images
    are assumed to be `stack_<idx>_0_0.tif
    """
    img_paths = []
    img_pathsA = []     # above
    img_pathsB = []     # below
    img_pathsE = []     # edge
    for idx in range(0,num_stacks):
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+10)+'_1_1.tif')
        img_pathsA = img_pathsA + glob.glob(pth+'/stack_'+str(idx+9)+'_1_1.tif')
        img_pathsB = img_pathsB + glob.glob(pth+'/stack_'+str(idx+11)+'_1_1.tif')
        img_pathsE = img_pathsE + glob.glob(epth+'/stack_'+str(idx+10)+'_1_1.tif')
    return img_paths, img_pathsA, img_pathsB, img_pathsE

def get_valid_img_stack(pth, epth, num_stacks):
    """
    Returns a list of training image paths. The training images
    are assumed to be `stack_<idx>_0_1.tif
    """
    img_paths = []
    img_pathsA = []     # above
    img_pathsB = []     # below
    img_pathsE = []     # edge
    for idx in range(0,num_stacks):
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+10)+'_2_2.tif')
        img_pathsA = img_pathsA + glob.glob(pth+'/stack_'+str(idx+9)+'_2_2.tif')
        img_pathsB = img_pathsB + glob.glob(pth+'/stack_'+str(idx+11)+'_2_2.tif')
        img_pathsE = img_pathsE + glob.glob(epth+'/stack_'+str(idx+10)+'_2_2.tif')

    return img_paths, img_pathsA, img_pathsB, img_pathsE




def apply_rotation(img1, seg1):
    """
    With 1/4th chance apply one of the following rotations,
    {0, 90, 180, 270} --> Counter clock wise
    """
    rotation_prob   = random.randint(0,4)
    rotation_degree = rotation_prob*90
    img1            = skimage.transform.rotate(img1, rotation_degree)
    seg1            = skimage.transform.rotate(seg1, rotation_degree)

    return img1, seg1, rotation_prob

def apply_horizontal_flip(img1, seg1):
    """
    With 1/2 the chance apply mirroring
    """
    flip_prob = random.randint(0,2)
    if flip_prob:
        img1 = img1[:,::-1]
        seg1 = seg1[:,::-1]
    return img1, seg1, flip_prob
        
def single_rotation(img1, rotation_prob):
    """
    With 1/4th chance apply one of the following rotations,
    {0, 90, 180, 270} --> Counter clock wise
    """    
    rotation_degree = rotation_prob*90
    img1            = skimage.transform.rotate(img1, rotation_degree)    

    return img1

def single_horizontal_flip(img1, flip_prob):
    """
    With 1/2 the chance apply mirroring
    """    
    if flip_prob:
        img1 = img1[:,::-1]
        
    return img1
        
def FixZeroLabels(seg):
    seg0 = (seg == 0)
    seg0 = seg0.astype(np.single)    

    minSeg = np.min(seg)
    maxSeg = np.max(seg)
    #print(" Range  " + str(minSeg) + " -> " + str(maxSeg))
    
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(seg0, structure)
    
    labeled[labeled > 0] = labeled[labeled > 0] + maxSeg
    nseg = seg + labeled

    #ScaleAndShow(nseg, 1)
    return nseg

def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')
    plt.show()


##############################################################
## Basic Training Program
if __name__ == '__main__':
    verbose = 0
    numEpochs = 1000
    num_stacks = 10
    THRESH_OFFSET = 0.75
    learningRate = 1.0
    learningRateGamma = 0.7
    random.seed(10)
    torch.manual_seed(10)
    
    # Setting up U-net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(in_channels=2, n_classes=1, depth=5, wf=8, padding=True, batch_norm=True, up_mode='upsample').to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # was
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    #optimizer = optim.Adadelta(model.parameters(), lr=learningRate)
    scheduler = StepLR(optimizer, step_size=100, gamma=learningRateGamma)

    # Paths for images and segmentation labels
    #trn_img_path = "/home/vj/Dropbox/LosAlamos/dsn/data/snemi3d/train-input_128x128"
    #trn_seg_path = "/home/vj/Dropbox/LosAlamos/dsn/data/snemi3d/train-labels_128x128"
    #trn_img_path = "d:\\image_data\\snemi3d\\train-input_128x128"
    #trn_edge_path = "d:\\image_data\\snemi3d\\train-membranes-idsia_128x128"
    #trn_seg_path = "d:\\image_data\\snemi3d\\train-labels_128x128"
    trn_img_path = "d:\\image_data\\snemi3d\\train-input_256x256"
    trn_edge_path = "d:\\image_data\\snemi3d\\train-membranes-idsia_256x256"
    trn_seg_path = "d:\\image_data\\snemi3d\\train-labels_256x256"
    #trn_img_path = "c:\\data\\snemi3d\\train-input_128x128"
    #trn_edge_path = "c:\\data\\snemi3d\\train-membranes-idsia_128x128"
    #trn_seg_path = "c:\\data\\snemi3d\\train-labels_128x128"


    # Training/Validation images
    (all_img_paths, allA, allB, allE) = get_train_img_stack(trn_img_path, trn_edge_path, num_stacks)
    all_seg_paths = get_train_img_paths(trn_seg_path, num_stacks)
    (val_img_paths, valA, valB, valE) = get_valid_img_stack(trn_img_path, trn_edge_path, num_stacks)
    val_seg_paths = get_valid_img_paths(trn_seg_path, num_stacks)

    all_img_paths.extend(val_img_paths) 
    all_seg_paths.extend(val_seg_paths)
    allA.extend(valA)
    allB.extend(valB)
    allE.extend(valE)

    mapIndexPosition = list(zip(all_img_paths, allA, allB, allE, all_seg_paths))
    #print(mapIndexPosition)
    random.shuffle(mapIndexPosition)
    (all_img_paths, allA, allB, allE, all_seg_paths) = zip(*mapIndexPosition)
    
    all_img_paths = list(all_img_paths)
    all_seg_paths = list(all_seg_paths)
    allA = list(allA)
    allB = list(allB)
    allE = list(allE)
    
    N2 = int(len(all_img_paths))    
    N = int(len(all_img_paths)/2)
    

    if True:
        trn_img_paths = all_img_paths[0:N]    
        val_img_paths = all_img_paths[N:N2]    
        trn_seg_paths = all_seg_paths[0:N]
        val_seg_paths = all_seg_paths[N:N2]
        trnA = allA[0:N]
        valA = allA[N:N2]
        trnB = allB[0:N]
        valB = allB[N:N2]
        trnE = allE[0:N]
        valE = allE[N:N2]
    else:
        val_img_paths = all_img_paths[0:N]    
        trn_img_paths = all_img_paths[N:N2]    
        val_seg_paths = all_seg_paths[0:N]
        trn_seg_paths = all_seg_paths[N:N2]
        valA = allA[0:N]
        trnA = allA[N:N2]
        valB = allB[0:N]
        trnB = allB[N:N2]
        valE = allE[0:N]
        trnE = allE[N:N2]

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

    for cepoch in range(0,numEpochs):
        print("Epoch :    ",str(cepoch))

        model.train()
        with torch.enable_grad():                            

            # Bar
            bar = progressbar.ProgressBar(maxval=len(trn_img_paths),\
                                        widgets=[progressbar.Bar('=', '    trn[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            # Training
            loss_lst_epoch   = []
            rerror_lst_epoch = []
            for trn_idx in range(0, len(trn_img_paths)):
                bar.update(trn_idx+1)
                trn_img_path = trn_img_paths[trn_idx]
                trn_imgE = trnE[trn_idx]
                trn_seg_path = trn_seg_paths[trn_idx]

                # Training image
                img1_o      = skio.imread(trn_img_path)
                #img1_o      = img1_o[0:128, 0:128]
                img1_e      = skio.imread(trn_imgE)
                #img1_e      = img1_e[0:128, 0:128]
                seg1_o      = skio.imread(trn_seg_path)
                #seg1_o      = seg1_o[0:128, 0:128]
                seg1_o      = FixZeroLabels(seg1_o)

                # With 1/4th chance choose a rotation, {0, 90, 180, 270}
                img1_r, seg1_r, prob = apply_rotation(img1_o, seg1_o)
                img1_re = single_rotation(img1_e, prob) 

                # With 1/2 chance choose to flip image left to right
                img1, seg1, prob = apply_horizontal_flip(img1_r, seg1_r)
                img1e = single_horizontal_flip(img1_re, prob) 

                Ysyn = np.expand_dims(seg1, axis=0)
                Xsyn = np.zeros((2, img1.shape[0], img1.shape[1]))        
                Xsyn[0,:,:] = img1
                Xsyn[1,:,:] = img1e
                Xsyn = np.expand_dims(Xsyn, axis=0)

                Xsyn = Xsyn.astype(np.single)
                Ysyn = Ysyn.astype(np.single)

                X4 = torch.tensor(Xsyn, requires_grad=False)                
                YL = Ysyn[0]    

                optimizer.zero_grad()
                
                X4 = X4.to(device)                         
                uOut = model(X4)
                loss, randError = RandLossDSN(uOut, YL, trn_idx)
                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]

                # Don't change model on last pass so that train and validation are aligned
                if trn_idx < len(trn_img_paths)-1:
                    loss.backward()
                    optimizer.step()

            # Finish bar
            bar.finish()
            trn_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            trn_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
            trn_rerror_lst     = trn_rerror_lst + [trn_cepoch_rerror]
            trn_loss_lst       = trn_loss_lst   + [trn_cepoch_loss]

            if trn_cepoch_loss < trn_best_loss:
                # Saving best loss model
                trn_best_loss       = trn_cepoch_loss
                trn_best_loss_epoch = cepoch
                trn_best_loss_model = model
                print("\t\tTraining loss   : ", str(trn_best_loss))                
                torch.save(trn_best_loss_model.state_dict(), "trn_loss_model_1.pth")

            if trn_cepoch_rerror < trn_best_rerror:
                # Saving best rerror model
                trn_best_rerror       = trn_cepoch_rerror
                trn_best_rerror_epoch = cepoch
                trn_best_rerror_model = model
                print("\t\tTraning error    : ", str(trn_best_rerror))                
                torch.save(trn_best_rerror_model.state_dict(), "trn_error_model_1.pth")

        
        # Validation every epoch        
        loss_lst_epoch   = []
        rerror_lst_epoch = []
        # Bar
        model.eval()
        with torch.no_grad():                

            bar = progressbar.ProgressBar(maxval=len(trn_img_paths),\
                                    widgets=[progressbar.Bar('-', '    Val[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for val_idx in range(0, len(val_img_paths)):
                bar.update(val_idx+1)
                # print("\t Validating on ",str(val_idx)," image")
                val_img_path = val_img_paths[val_idx]
                val_imgE = valE[val_idx]

                val_seg_path = val_seg_paths[val_idx]

                # Validation image
                img1_o      = skio.imread(val_img_path)
                #img1      = img1_o[0:128, 0:128]            
                img1e_o      = skio.imread(val_imgE)
                #img1e      = img1e_o[0:128, 0:128]                
                seg1      = skio.imread(val_seg_path)
                #seg1      = seg1[0:128, 0:128]
                seg1      = FixZeroLabels(seg1)

                #img1        = np.zeros((img1_o.shape[0], img1_o.shape[1]))  
                #img1e        = np.zeros((img1e_o.shape[0], img1e_o.shape[1]))  
                # for ri in range(4): 
                #     for fi in range(2): 
                #         img1_r = single_rotation(img1_o, ri) 
                #         img1e_r = single_rotation(img1e_o, ri) 
                #         img1_r = single_horizontal_flip(img1_r, fi) 
                #         img1e_r = single_horizontal_flip(img1e_r, fi) 

                #         img1 = img1 + img1_r
                #         img1e = img1e + img1e_r

                # img1 = img1 / 8
                # img1e = img1e / 8

                Ysyn = np.expand_dims(seg1, axis=0)
                Xsyn = np.zeros((2, img1.shape[0], img1.shape[1]))        
                Xsyn[0,:,:] = img1
                Xsyn[1,:,:] = img1e

                Xsyn = np.expand_dims(Xsyn, axis=0)

                Xsyn = Xsyn.astype(np.single)
                Ysyn = Ysyn.astype(np.single)

                X4 = torch.tensor(Xsyn, requires_grad=False)                
                YL = Ysyn[0]    

                X4 = X4.to(device)                         
                uOut = model(X4)

                loss, randError = RandLossDSN(uOut, YL, cepoch)
                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch       + [loss.detach().numpy().tolist()]

            # Finish bar
            bar.finish()
            scheduler.step()

            val_cepoch_loss    = sum(loss_lst_epoch)/len(loss_lst_epoch)
            val_cepoch_rerror  = sum(rerror_lst_epoch)/len(rerror_lst_epoch)
            val_rerror_lst     = val_rerror_lst + [val_cepoch_rerror]
            val_loss_lst       = val_loss_lst   + [val_cepoch_loss]

            if val_cepoch_loss < val_best_loss:
                # Saving best loss model
                val_best_loss       = val_cepoch_loss
                val_best_loss_epoch = cepoch
                val_best_loss_model = model
                print("\t\tBest val loss ", str(val_best_loss))
                torch.save(val_best_loss_model.state_dict(), "val_loss_model_1.pth")

            if val_cepoch_rerror < val_best_rerror:
                # Saving best rerror model
                val_best_rerror       = val_cepoch_rerror
                val_best_rerror_epoch = cepoch
                val_best_rerror_model = model
                print("\t\tBest val error ", str(val_best_rerror))
                torch.save(val_best_rerror_model.state_dict(), "val_error_model_1.pth")

    # Saving all the losses and rand_errors
    with open('trn_loss_1.pkl', 'wb') as f:
        pickle.dump(trn_loss_lst, f)
    with open('trn_error_1.pkl', 'wb') as f:
        pickle.dump(trn_rerror_lst, f)
        
    with open('val_loss_1.pkl', 'wb') as f:
        pickle.dump(val_loss_lst, f)
    with open('val_errors_1.pkl', 'wb') as f:
        pickle.dump(val_rerror_lst, f)
        
    print("\nFinal best training loss", str(trn_best_loss))
    print("Final best traning rand error", str(trn_best_rerror))    
    print("Final best validation loss", str(val_best_loss))
    print("Final best validation rand error", str(val_best_rerror))

    # Loading model
    # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
    # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
    # uOut1 = model(X3)
    # uOut2 = loaded_model(X3)
        
                

 #Traning error    :  0.02950925409651963, 0.026672635244041083, 0.03196152544629914
 #Validation error :  0.049092329027946274, 0.05092111785558965, 0.04836977232377884
 #  




