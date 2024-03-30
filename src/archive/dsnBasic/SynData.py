import numpy as np

def GenSyn1():
    img = np.zeros((256, 256), np.single)
    
    for j in range(127):
        for i in range(127):
            img[i+64, j+64] = j+10

    return img
