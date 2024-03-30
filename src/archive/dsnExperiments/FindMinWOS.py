import progressbar
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pdb
import math
import random
from operator import itemgetter




##############################################################
## Find min wos
if __name__ == '__main__':

    verbose = 0
    theSeed = 0
    random.seed(theSeed)
    torch.manual_seed(theSeed)

    w = [0.3, 0.35, 0.4]
    b = 0.6
    N = 3

    j = 1


    indices, wSorted = zip(*sorted(enumerate(wSorted), key=itemgetter(1)))

    print(indicies)
    #list(L_sorted)
    #list(indices)
    #math.factorial
    #L = [2,3,1,4,5]