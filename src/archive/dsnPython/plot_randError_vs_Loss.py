"""
The following script plots Loss Rand errors for training and
Validation

Usage
-----
python plot_randError_vs_Loss.py <rand error pkl> <loss pkl> "<title>,<xlabel>,<ylabel>"

Example
-------
python plot_randError_vs_Loss.py trn_rerrors.pkl trn_losses.pkl "RandError Vs Loss, Epochs, RandError Vs Normalized loss"

"""
import os
import pdb
import sys
import glob
import pickle
import matplotlib.pyplot as plt
from scipy import stats


# Check for arguments
if not(len(sys.argv) == 4):
    print('USAGE:\n'+\
          '\tpython plot_randError_vs_Loss.py <rand error pkl> <loss pkl> "<title>,<xlabel>,<ylabel>"\n'+\
          '\tpython plot_randError_vs_Loss.py trn_rerrors.pkl trn_losses.pkl "RandError Vs Loss, Epochs, RandError Vs Normalized loss"\n')
    sys.exit()

# Loading pickle files
rerr_pkl      = pickle.load(open(sys.argv[1], 'rb'))
loss_pkl      = pickle.load(open(sys.argv[2], 'rb'))
norm_loss_pkl = [x/max(loss_pkl) for x in loss_pkl]
norm_rerr_pkl = [x/max(rerr_pkl) for x in rerr_pkl]

# T Test
"""
This is a two-sided test for the null hypothesis that 2 independent samples 
have identical average (expected) values. 
This test assumes that the populations have identical variances by default.
"""
t_statistic, p_value = stats.ttest_ind(norm_loss_pkl, norm_rerr_pkl)
print("P Value:", p_value)
print("T Stat :", t_statistic)

# Plotting
fig, ax = plt.subplots()
ax.plot(norm_rerr_pkl, color='blue', label='Rand Error')
ax.plot(norm_loss_pkl, color='orange', label='Loss')

# Fonts and titles
plt_info  = sys.argv[3]
plt_title = plt_info.split(',')[0]
x_label   = plt_info.split(',')[1]
y_label   = plt_info.split(',')[2]

fig.suptitle(plt_title, fontsize='32')
ax.set_ylabel(y_label, fontsize='25')
ax.set_xlabel(x_label, fontsize='25')

ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,
          fontsize='25')

# Show plot
plt.show()






    
