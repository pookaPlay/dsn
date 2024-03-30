import sys
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
#
import json

dateToString = "%Y-%b-%d %H:%M:%S"
dateToStringFrac = "%Y-%b-%d %H:%M:%S"
myDateFmt = mdates.DateFormatter('%H:%M:%S')


linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


if __name__ == '__main__':

    with open('radius_errors.pkl', 'rb') as f:
        threshError = pickle.load(f)

    print(threshError)

    #mpl.rc('lines', linewidth=4, linestyle='-.')
    labels = ['R=1', 'R=2', 'R=3', 'R=4', 'R=5']

    myThresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    axes = plt.gca()
    #axes.set_xlim([xmin,xmax])
    axes.set_ylim([0.0,0.2])

    plt.xlabel('Offset Value', fontsize=18)
    plt.ylabel('Rand Error', fontsize=18)
    
    #"solid", "dotted", "dashed" or "dashdot"
    #line1, = ax.plot(x, y, label='Using set_dashes()')
    #line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    #0, (1, 10))
    #(0, (5, 10))
    colors = ['r', 'g', 'b', 'k', 'c']
    linestyle = ['-', '--', '-.', ':', ':']
    
    lines = list()
    for i in range(len(threshError)):
        line = ax.plot(myThresh, threshError[i], linestyle=linestyle[i], linewidth=1.5, color=colors[i], label=labels[i])
        lines.append(line)
    plt.legend(loc="upper left")
    #plt.legend(handles=[lines[0], lines[1], lines[2], lines[3], lines[4]], ['r', 'r', 'g', 'h', 'h'])
    #ax.set(xlabel='Offset Value', ylabel='Rand Error', titlesize=16) 
    #title='Constrained correlation clustering result as a function of offset value')
    
    fig.savefig("offset_errors_verse_radius.png")
    plt.show()
    
    #flata = np.array(amplitudeList).flatten()
    #newt = list()
    #colors = list()
    #colors.append('blue')

    #axFP.scatter(newt, f1, c=colors, marker='+')
    #axFP.xaxis.set_major_formatter(myDateFmt); 
    #axFP.set_xlim(newt[0], newt[-1])

     #axFP.set_title('1D') 
     #flata = np.array(amplitudeList).flatten()
     # #plt.yticks(timeList)
        #axFP.set_ylabel('time (' + str(timeFactor) + ' sec)')
        #axFP.set_xlim(newt[0], newt[-1])
        #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
        #plt.show()
    