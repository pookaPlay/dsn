"""
The following script has code supports plots for various figure for paper.

    0
        Two band rerrors with scaled training loss.
    1
        ???

Usage
-----
python plots_for_paper.py <plot id>

Example
-------
python plots_for_paper.py 0
"""
import os
import pdb
import sys
import glob
import pickle
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # Read files
    #trnError1 = pickle.load(open('train_error_purity0_final.pkl','rb'))
    #trnError2 = pickle.load(open('train_error_equal0_final.pkl','rb'))
    trnError1 = pickle.load(open('train_purity0_final.pkl','rb'))
    trnError2 = pickle.load(open('train_equal0_final.pkl','rb'))

    # Print best loss and rerr    
    print("Training Error Purity: ", str(min(trnError1)))
    print("Training Error Equal : ", str(min(trnError2)))
    
    winSize = 10
    error1 = []
    error2 = []

    for i in range(winSize, len(trnError1)):
        avg1 = sum(trnError1[(i-winSize):i])/winSize
        avg2 = sum(trnError2[(i-winSize):i])/winSize
        error1.append(avg1)
        error2.append(avg2)
    #max_loss = max(trn_loss_pkl)
    #trn_loss_scaled = [x/max_loss for x in trn_loss_pkl]

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(error1, color='blue', linestyle='-', label='Training Purity')
    ax.plot(error2, color='green', linestyle='-', label='Training Equal')    
    ax.set_ylim([0,0.5])

    # Fonts and titles
    fig.suptitle('Training Errors', fontsize='32')

    ax.set_ylabel('Rand Error', fontsize='25')
    ax.set_xlabel('Epochs', fontsize='25')

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,
              fontsize='20')

    plt.show()
