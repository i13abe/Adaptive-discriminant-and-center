import numpy as np
import matplotlib.pyplot as plt

def histgram(data, bins = 10, ylim = 500, filename = "histgram_test"):
    classes = len(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.hist(data, bins = bins)
    
    ax.set_xlabel('x')
    ax.set_ylabel('freq')
    ax.set_ylim(0, ylim)
    plt.savefig(filename)
    plt.close()
    