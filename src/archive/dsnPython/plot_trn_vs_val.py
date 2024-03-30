"""
The following script plots Loss Rand errors for training and
Validation

Usage
-----
python plot_runs.py <Training pkl> <Validation pkl> <Training "<title>,<xlabel>,<ylabel>"

Example
-------
python plot_runs.py Four_band_trn_rerrors.pkl Four_band_val_rerrors.pkl "Four channels,Epochs,Rand Error"
"""
import os
import pdb
import sys
import glob
import pickle
import matplotlib.pyplot as plt


# Check for arguments
if not(len(sys.argv) == 4):
    print('USAGE:\n'+\
          '\tpython plot_runs.py <Training pkl> <Validation pkl>\n'+\
          '\tpython plot_runs.py Four_band_trn_rerrors.pkl '+\
          'Four_band_val_rerrors.pkl "Four channels"')
    sys.exit()

# Loading pickle files
trn_pkl   = pickle.load(open(sys.argv[1], 'rb'))
val_pkl   = pickle.load(open(sys.argv[2], 'rb'))

# Plotting
fig, ax = plt.subplots()
ax.plot(trn_pkl, color='blue', label='Training')
ax.plot(val_pkl, color='orange', label='Validation')

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






    
