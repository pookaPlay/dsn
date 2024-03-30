import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
from unet import UNet
import QuanSynData
import BSDSData as bsds
import numpy as np
import matplotlib.pyplot as plt
import SegEval as ev
import networkx as nx
import pickle
import pdb
import skimage.io as skio
from scipy.signal import medfilt as med_filt
import math
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from scipy.ndimage.measurements import label

# MATPLOTLIB defaults
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15


def SetupGraph(npGXY, W, H, radius, nodeLabels):    

    # Setup input graph 
    #G = nx.grid_2d_graph(W, H)

    G = nx.Graph()
    for v in range(H): 
        for u in range(W):
            G.add_node((u,v))
    
    for r in radius:
        for v in range(H): 
            for u in range(W):
                if u > r-1 and v > r-1:
                    G.add_edge((u,v), (u-r,v-r))
                if u < W-r and v < H-r:
                    G.add_edge((u,v), (u+r,v+r))
                if u > r-1 and v < H-r:
                    G.add_edge((u,v), (u-r,v+r))
                if u < W-r and v > r-1:
                    G.add_edge((u,v), (u+r,v-r))
                if u > r-1:
                    G.add_edge((u,v), (u-r,v))
                if v > r-1:
                    G.add_edge((u,v), (u,v-r))
                if u < W-r:
                    G.add_edge((u,v), (u+r,v))
                if v < H-r:
                    G.add_edge((u,v), (u,v+r))

    #num = G.number_of_edges()
    #print("Graph Edges: " + str(num))

    for u, v, d in G.edges(data = True):
        d['weight'] =  (npGXY[0, u[0], u[1]] + npGXY[0, v[0], v[1]])/2.0
    
    if nodeLabels.any() == None:
        return(G,0)
    
    # Setup labels
    nlabels_dict = dict()
    for u, v, d in G.edges(data = True):
        nlabels_dict[u] = nodeLabels[u[0], u[1]]
        nlabels_dict[v] = nodeLabels[v[0], v[1]]
            
    return (G, nlabels_dict)

############################################################################################################################
############################################################################################################################
############################################################################################################################
## DSN Inference Methods
############################################################################################################################
############################################################################################################################


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

    (G, temp) = SetupGraph(npGXY, W, H, None)

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

def RandLossDSN(uOut, nodeLabels, trn_idx, radius):
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
    (G, nlabels_dict) = SetupGraph(npGXY, W, H, radius, nodeLabels)    
            
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

    # print("\t\t" + str(trn_idx) + " Loss " + str(randLoss.item()) + "  and Rand " + str(randError))
    return(randLoss, randError)    


def get_all_img_paths(pth, idxs):
    """
    Returns paths of all images having particular index
    """
    img_paths = []
    for idx in idxs:
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+20)+'_5_4.tif')

    return img_paths



def get_train_img_paths(pth, num_stacks):
    """
    Returns a list of training image paths. The training images
    are assumed to be `stack_<idx>_0_0.tif
    """
    img_paths = []
    for idx in range(0,num_stacks):
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+20)+'_5_3.tif')
    return img_paths

def get_valid_img_paths(pth, num_stacks):
    """
    Returns a list of training image paths. The training images
    are assumed to be `stack_<idx>_0_1.tif
    """
    img_paths = []
    for idx in range(0,num_stacks):
        img_paths = img_paths + glob.glob(pth+'/stack_'+str(idx+20)+'_5_5.tif')

    return img_paths

   
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

##############################################################
## Basic Training Program
if __name__ == '__main__':
    verbose = 0
    num_stacks = 10
    THRESH_OFFSET = 0.65

    torch.manual_seed(0)
    # Setting up U-net
    device = 'cpu'

    # Paths for images and segmentation labels
    #trn_img_path = "/home/vj/Dropbox/LosAlamos/dsn/data/snemi3d/train-input_128x128"
    #trn_seg_path = "/home/vj/Dropbox/LosAlamos/dsn/data/snemi3d/train-labels_128x128"
    trn_img_path = "d:\\image_data\\snemi3d\\train-membranes-idsia_128x128"
    trn_seg_path = "d:\\image_data\\snemi3d\\train-labels_128x128"

    # Training/Validation images
    trn_img_paths = get_train_img_paths(trn_img_path, num_stacks)
    trn_seg_paths = get_train_img_paths(trn_seg_path, num_stacks)
    #trn_img_paths = get_valid_img_paths(trn_img_path, num_stacks)
    #trn_seg_paths = get_valid_img_paths(trn_seg_path, num_stacks)

    # Initializing training and validation lists to be empty
    trn_loss_lst   = []
    trn_rerror_lst = []
    val_loss_lst   = []
    val_rerror_lst  = []

    # Initializing best loss and random error
    trn_best_loss   = math.inf
    trn_best_rerror = math.inf
    val_best_loss   = math.inf
    val_best_rerror = math.inf

    allError = list()
    
    for gi in range(1, 6, 1):
        # plot 1        
        radius = [gi]

        # plot 2
        radius = range(1,gi+1)
        #radius = list()
        #if gi == 1:
        #    radius = [1]
        #else:
        #    radius = range(1, (gi-1)*5+1, 5)

        print("Radius: " + str(radius))

        threshError = list()
        #myThresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        myThresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #myThresh = [0.65]
        #for THRESH_OFFSET in range(0.5, 0.95, 0.01):
        for THRESH_OFFSET in myThresh:        
            print("Threshold: " + str(THRESH_OFFSET))

            # Bar
            bar = progressbar.ProgressBar(maxval=len(trn_img_paths),\
                                            widgets=[progressbar.Bar('=', '    test[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            loss_lst_epoch   = []
            rerror_lst_epoch = []
            for trn_idx in range(0, len(trn_img_paths)):
                bar.update(trn_idx+1)
                trn_img_path = trn_img_paths[trn_idx]
                trn_seg_path = trn_seg_paths[trn_idx]
                #print("\t Training on ",str(trn_img_path)," image")
                # Training image
                img1      = skio.imread(trn_img_path)
                img1      = img1[0:128, 0:128]
                seg1      = skio.imread(trn_seg_path)
                seg1      = seg1[0:128, 0:128]
                seg1      = FixZeroLabels(seg1)

                Ysyn = np.expand_dims(seg1, axis=0)
                Xsyn = np.zeros((1, img1.shape[0], img1.shape[1]))        
                Xsyn[0,:,:] = img1
                Xsyn = np.expand_dims(Xsyn, axis=0)

                Xsyn = Xsyn.astype(np.single)
                Ysyn = Ysyn.astype(np.single)

                XS1 = np.expand_dims(Xsyn[0], axis=0)
                YL = Ysyn[0]    

                #(fracSame, sameCount, diffCount) = AnalyzeGroundTruth(YL)
                #print("Ground Truth Graph Ratio: " + str(fracSame) + "     #same/#diff: " + str(sameCount) + "/" + str(diffCount))

                X1 = torch.tensor(XS1, requires_grad=False)            
                uOut = X1
                loss, randError = RandLossDSN(uOut, YL, trn_idx, radius)
                rerror_lst_epoch  = rerror_lst_epoch + [randError]
                loss_lst_epoch    = loss_lst_epoch + [loss.detach().numpy().tolist()]

            avgloss = sum(loss_lst_epoch)/len(loss_lst_epoch)
            avgerror = sum(rerror_lst_epoch)/len(rerror_lst_epoch)

            bar.finish()
            print("Averga loss : ", str(avgloss))
            #print(loss_lst_epoch)
            print("Averga error:", str(avgerror))
            #print(rerror_lst_epoch)
            threshError.append(avgerror)

        
        allError.append(threshError)

    with open('skip_errors.pkl', 'wb') as f:
        pickle.dump(allError, f)

    #with open('thresh_errors.txt', 'w') as f:
    #print(threshError, file = f)

        # Loading model
        # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
        # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
        # uOut1 = model(X3)
        # uOut2 = loaded_model(X3)
