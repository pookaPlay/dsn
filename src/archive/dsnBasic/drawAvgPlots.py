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
import numpy as np
import math

if __name__ == '__main__':
    
    files1 = ['train_error_purity_final.pkl', 'train_error_purity2_final.pkl', 'train_error_purity3_final.pkl', 'train_error_purity4_final.pkl', 'train_error_purity5_final.pkl', 'train_error_purity6_final.pkl', 'train_error_purity7_final.pkl', 'train_error_purity8_final.pkl', 'train_error_purity9_final.pkl']
    files2 = ['train_error_equal_final.pkl', 'train_error_equal2_final.pkl', 'train_error_equal3_final.pkl', 'train_error_equal4_final.pkl', 'train_error_equal5_final.pkl', 'train_error_equal6_final.pkl', 'train_error_equal7_final.pkl', 'train_error_equal8_final.pkl', 'train_error_equal9_final.pkl']
    #files1 = ['train_error_purity2_final.pkl', 'train_error_purity3_final.pkl', 'train_error_purity4_final.pkl', 'train_error_purity5_final.pkl', 'train_error_purity6_final.pkl', 'train_error_purity7_final.pkl', 'train_error_purity8_final.pkl', 'train_error_purity9_final.pkl']
    #files2 = ['train_error_equal_final.pkl', 'train_error_equal2_final.pkl', 'train_error_equal4_final.pkl', 'train_error_equal5_final.pkl', 'train_error_equal6_final.pkl', 'train_error_equal7_final.pkl', 'train_error_equal8_final.pkl', 'train_error_equal9_final.pkl']

    winSize = 200
    error1 = []
    error2 = []
    var1 = []
    var2 = []

    allError1 = dict()
    allError2 = dict()

    for i in range(len(files1)):
        # Read files
        trnError1 = pickle.load(open(files1[i],'rb'))
        trnError2 = pickle.load(open(files2[i],'rb'))
        allError1[i] = list()
        allError2[i] = list()

        if i==0:
            for ii in range(winSize, 1000):
                avg1 = sum(trnError1[(ii-winSize):ii])/winSize
                avg2 = sum(trnError2[(ii-winSize):ii])/winSize
                #avg1 = min(trnError1[(ii-winSize):ii])
                #avg2 = min(trnError2[(ii-winSize):ii])

                allError1[i].append(avg1)
                allError2[i].append(avg2)
                error1.append(avg1)
                error2.append(avg2)
                var1.append(avg1*avg1)
                var2.append(avg2*avg1)

        else:
            for ii in range(winSize, 1000):
                avg1 = sum(trnError1[(ii-winSize):ii])/winSize
                avg2 = sum(trnError2[(ii-winSize):ii])/winSize            
                #avg1 = min(trnError1[(ii-winSize):ii])
                #avg2 = min(trnError2[(ii-winSize):ii])

                allError1[i].append(avg1)
                allError2[i].append(avg2)
            
                error1[ii-winSize] = error1[ii-winSize] + avg1
                error2[ii-winSize] = error2[ii-winSize] + avg2
                var1[ii-winSize] = var1[ii-winSize] + avg1*avg1
                var2[ii-winSize] = var2[ii-winSize] + avg2*avg2
                
    # Print best loss and rerr    
    for ii in range(len(error1)):
        error1[ii] = error1[ii] / len(files1)
        error2[ii] = error2[ii] / len(files1)
        var1[ii] = var1[ii] / len(files1)
        var2[ii] = var2[ii] / len(files1)
        var1[ii] = math.sqrt(var1[ii] - (error1[ii]*error1[ii]))
        var2[ii] = math.sqrt(var2[ii] - (error2[ii]*error2[ii]))

    # Plotting
    fig, ax = plt.subplots()
    #for i in range(len(files1)):   
    #    ax.plot(allError1[i], color='blue', linestyle=':', linewidth=0.5)
    #    ax.plot(allError2[i], color='green', linestyle=':', linewidth=0.5)    

    #ax.plot(error1, color='blue', linestyle='-', label='Training Purity', linewidth=2)    
    #ax.plot(error2, color='green', linestyle='-', label='Training Equal', linewidth=2)
    
    x = np.arange(len(error1))
    print(len(x))
    print(len(error1))
    sx = x[:-1:2]
    s1 = error1[:-1:2]
    v1 = var1[:-1:2]
    s2 = error2[:-1:2]
    v2 = var2[:-1:2]
    print(len(s1))    
    plt.errorbar(sx, s1, yerr=v1, color='blue', linestyle='--', errorevery=47, label='Purity weighted')
    plt.errorbar(sx, s2, yerr=v2, color='green', errorevery=49, label='Equally weighted')    
    ax.set_ylim([0,0.2])

    # Fonts and titles
    ax.set_ylabel('Rand Error', fontsize='25')
    ax.set_xlabel('Epochs', fontsize='25')

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,
              fontsize='20')

    plt.show()
