import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
from unet import UNet
import numpy as np
import matplotlib.pyplot as plt
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
import dsn


def ScaleAndShow(img, fignum):
    minv = np.min(img)
    maxv = np.max(img)
    print("Fig %i: Range %f -> %f" % (fignum, minv, maxv))
    plt.figure(fignum)
    simg = img - minv 
    if abs(maxv - minv) > 1e-4:
        simg = simg / (maxv - minv)
    
    plt.imshow(simg, cmap='gray')

def draw_boundary(oimg, limg):
    """
    Raster scans the image and when it detects a change (four connectivity)
    it marks currrent pixel as active.
    oimg: 
    -----
        Original image
    limg:
    -----
        Labeled image
    """
    ht,wd           = limg.shape
    norm_img        = oimg/oimg.max()
    norm_img_rgb    = np.zeros((ht, wd, 3))
    norm_img_rgb[:,:,0] = norm_img
    norm_img_rgb[:,:,1] = norm_img
    norm_img_rgb[:,:,2] = norm_img
    # Loop over labeled image leaving one pixel at boundaries
    #     1. Left to right
    #     2. Top to bottom
    for ridx in range(1,ht-1):
        for cidx in range (1,wd-1):
            cur_lab    = limg[ridx  , cidx]
            top_lab    = limg[ridx-1, cidx]
            bot_lab    = limg[ridx+1, cidx]
            lft_lab    = limg[ridx  , cidx-1]
            rgt_lab    = limg[ridx  , cidx+1]
            all_labels = np.asarray([cur_lab,
                                     top_lab, bot_lab,
                                     lft_lab, rgt_lab])
            num_uniq_labels = np.unique(all_labels)
            if len(num_uniq_labels) > 1:
                norm_img_rgb[ridx, cidx-1,0] = 0
                norm_img_rgb[ridx, cidx-1,1] = 1
                norm_img_rgb[ridx, cidx-1,2] = 0
                norm_img_rgb[ridx, cidx,0] = 0
                norm_img_rgb[ridx, cidx,1] = 1
                norm_img_rgb[ridx, cidx,2] = 0
    return norm_img_rgb
    
##############################################################
## Basic Training Program
if __name__ == '__main__':

    vname = "saved\\edge_result_62.pkl"
    with open(vname, 'rb') as f:
        edge = pickle.load(f)            

    ScaleAndShow(edge, 0)
    plt.show()
    # Loading model
    # loaded_model = UNet(in_channels=1, n_classes=1, depth=5, padding=True, up_mode='upsample').to(device)
    # loaded_model.load_state_dict(torch.load("best_val_model.pth"))
    # uOut1 = model(X3)
    # uOut2 = loaded_model(X3)
