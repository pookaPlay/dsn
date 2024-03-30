import os
import pdb
import sys
import glob
import skimage.io as skio
import numpy as np
import torch
# functions/methods from current module
import os
import skimage.transform
import pickle

from scipy.ndimage.measurements import label as SciLabel

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
    labeled, ncomponents = SciLabel(seg0, structure)
    
    labeled[labeled > 0] = labeled[labeled > 0] + maxSeg
    nseg = seg + labeled

    #ScaleAndShow(nseg, 1)
    return nseg


def GetAllSave():
    theSeed = 0
    
    torch.manual_seed(theSeed)
    np.random.seed(theSeed)

    dataDir = "d:\\image_data\\snemi3d\\"
    extractDir = "d:\\image_data\\snemi3d-extracts\\"

    imgName = dataDir + "train-input.tif"
    labelName = dataDir + "train-labels.tif"
    edgeName = dataDir + "train-membranes-idsia.tif"
    
    width = 1024
    height = 1024
    depth = 100
    
    cutsz = 512    
    numBands = 50
    numSpatial = 3
    dataAugment = False

    img = skio.imread(imgName)
    edge = skio.imread(edgeName)
    labeli = skio.imread(labelName)

    trainName0 = extractDir + "train512_0." + str(theSeed) + ".pkl"
    trainName1 = extractDir + "train512_1." + str(theSeed) + ".pkl"
    trainName2 = extractDir + "train512_2." + str(theSeed) + ".pkl"
    trainName4 = extractDir + "train512_4." + str(theSeed) + ".pkl"
    trainNameLabel = extractDir + "trainLabel512." + str(theSeed) + ".pkl"

    validName0 = extractDir + "valid512_0." + str(theSeed) + ".pkl"
    validName1 = extractDir + "valid512_1." + str(theSeed) + ".pkl"
    validName2 = extractDir + "valid512_2." + str(theSeed) + ".pkl"
    validName4 = extractDir + "valid512_4." + str(theSeed) + ".pkl"
    validNameLabel = extractDir + "validLabel512." + str(theSeed) + ".pkl"

    bands = np.random.permutation(numBands) + 1
    #spatial = np.random.permutation(numSpatial)
    spatial = [0, 1, 2]

    numTrain = int((numBands * numSpatial)/2)
    numValid = int((numBands * numSpatial)/2)
    print(len(bands))
    print("Generating " + str(numTrain) + " training and " + str(numValid) + " validation")
    
    train0 = np.zeros((numTrain, 1, cutsz, cutsz)) # edge
    train1 = np.zeros((numTrain, 1, cutsz, cutsz))     # data  
    train2 = np.zeros((numTrain, 2, cutsz, cutsz))    # data + edge
    train4 = np.zeros((numTrain, 4, cutsz, cutsz))    # 3data + edge
    trainLabel = np.zeros((numTrain, 1, cutsz, cutsz)) 

    valid0 = np.zeros((numValid, 1, cutsz, cutsz))    # edge
    valid1 = np.zeros((numValid, 1, cutsz, cutsz))    # data  
    valid2 = np.zeros((numValid, 2, cutsz, cutsz))    # data + edge
    valid4 = np.zeros((numValid, 4, cutsz, cutsz))    # 3data + edge
    validLabel = np.zeros((numValid, 1, cutsz, cutsz)) 

    upto = 0
    traini = 0
    validi = 0
    for b in bands:
        
        label = FixZeroLabels(labeli[b])

        for s in spatial:
            if s == 0:
                x = 1
                y = 0
            elif s == 1:
                x = 0
                y = 1
            else:
                x = 1
                y = 1

            #x = int(np.floor(s / 4))
            #y = s % 4
            row_st     = x * cutsz
            row_en     = (x + 1) * cutsz
            col_st     = y * cutsz
            col_en     = (y + 1) * cutsz
            
            imgChip = img[b, row_st:row_en, col_st:col_en]
            labelChip = label[row_st:row_en, col_st:col_en]
            edgeChip = edge[b, row_st:row_en, col_st:col_en]

            
            if (upto % 2) == 0:  # Train
                print("Train: " + str(b) + " at " + str(x) + "," + str(y))
                if dataAugment:
                    pass
                    # With 1/4th chance choose a rotation, {0, 90, 180, 270}
                    #img1_r, seg1_r, prob = apply_rotation(img1_o, seg1_o)
                    #img1_re = single_rotation(img1_e, prob) 

                    # With 1/2 chance choose to flip image left to right
                    #img1, seg1, prob = apply_horizontal_flip(img1_r, seg1_r)
                    #img1e = single_horizontal_flip(img1_re, prob) 
                train0[traini, 0, :, :] = edgeChip
                train1[traini, 0, :, :] = imgChip
                train2[traini, 0, :, :] = edgeChip
                train2[traini, 1, :, :] = imgChip
                train4[traini, 0, :, :] = edgeChip                
                train4[traini, 1, :, :] = img[b-1, row_st:row_en, col_st:col_en]
                train4[traini, 2, :, :] = imgChip
                train4[traini, 3, :, :] = img[b+1, row_st:row_en, col_st:col_en]
                trainLabel[traini, 0, :, :] = labelChip
                traini = traini + 1
            else:  #valid
                print("Valid: " + str(b) + " at " + str(x) + "," + str(y))
                valid0[validi, 0, :, :] = edgeChip
                valid1[validi, 0, :, :] = imgChip
                valid2[validi, 0, :, :] = edgeChip
                valid2[validi, 1, :, :] = imgChip
                valid4[validi, 0, :, :] = edgeChip                
                valid4[validi, 1, :, :] = img[b-1, row_st:row_en, col_st:col_en]
                valid4[validi, 2, :, :] = imgChip
                valid4[validi, 3, :, :] = img[b+1, row_st:row_en, col_st:col_en]
                validLabel[validi, 0, :, :] = labelChip
                validi = validi + 1

            upto = upto + 1

    #Xsyn = Xsyn.astype(np.single)
    #Ysyn = Ysyn.astype(np.single)

    with open(trainName0, 'wb') as f:
        pickle.dump(train0, f)
    with open(trainName1, 'wb') as f:
        pickle.dump(train1, f)
    with open(trainName2, 'wb') as f:
        pickle.dump(train2, f)
    with open(trainName4, 'wb') as f:
        pickle.dump(train4, f)
    with open(trainNameLabel, 'wb') as f:
        pickle.dump(trainLabel, f)

    with open(validName0, 'wb') as f:
        pickle.dump(valid0, f)
    with open(validName1, 'wb') as f:
        pickle.dump(valid1, f)
    with open(validName2, 'wb') as f:
        pickle.dump(valid2, f)
    with open(validName4, 'wb') as f:
        pickle.dump(valid4, f)
    with open(validNameLabel, 'wb') as f:
        pickle.dump(validLabel, f)


if __name__ == '__main__':
    
    GetAllSave()

