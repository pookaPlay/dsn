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




def plot_index_0():
    """
    Two band rand errors with sacaled training loss
    """
    # Read files
    trn_rerr_pkl = pickle.load(open('trn_rerrors_2_channel.pkl','rb'))
    val_rerr_pkl = pickle.load(open('val_rerrors_2_channel.pkl','rb'))
    trn_loss_pkl = pickle.load(open('trn_losses_2_channel.pkl', 'rb'))

    # Print best loss and rerr
    print("Best Training loss: ", str(min(trn_loss_pkl)))
    print("Best Training rerr: ", str(min(trn_rerr_pkl)))
    print("Best Validation rerr: ", str(min(val_rerr_pkl)))

    # Scaling training loss to [0,1]
    max_loss = max(trn_loss_pkl)
    trn_loss_scaled = [x/max_loss for x in trn_loss_pkl]

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(trn_rerr_pkl, color='blue', linestyle='-', label='Training Rand Error')
    ax.plot(val_rerr_pkl, color='orange', linestyle=':', label='Validation Rand Error')
    ax.plot(trn_loss_scaled, color='green', linestyle='--', label='Training Loss scaled to [0,1]')
    ax.set_ylim([0,1])

    # Fonts and titles
    fig.suptitle('Two channel DSN', fontsize='32')

    ax.set_ylabel('Rand Error and Scaled Loss', fontsize='25')
    ax.set_xlabel('Epochs', fontsize='25')

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,
              fontsize='20')
    

    


if __name__ == '__main__':
    
    # Check for arguments
    if not(len(sys.argv) == 2):
        print('USAGE:\n'+\
              '\tpython plots_for_paper.py <plot id>\n'+\
              '\tpython plots_for_paper.py 0')
        sys.exit()

    plt_id = int(sys.argv[1])

    if plt_id == 0:
        fig = plot_index_0()
    else:
        print('ERROR: Unsupported plot index')




# Show plot
plt.show()
