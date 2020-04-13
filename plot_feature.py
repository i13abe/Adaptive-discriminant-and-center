from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm

class PlotFeature(object):
    def __init__(self, based_labels, colors = None):
        self.based_labels = based_labels
        
        self.colors = colors
            
    def plot_feature(self, data, labels, center, xlim = None, ylim = None, filename = "test"):
        
        plt.figure(figsize=(6,6))
        num = len(center)
    
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        if ylim:
            plt.ylim(ylim[0], ylim[1])
            
        for i in range(num):
            inds = np.where(labels == i)[0]
            if self.colors is None:
                plt.scatter(data[inds,0], data[inds,1], alpha = 0.5)
            else:
                plt.scatter(data[inds,0], data[inds,1], alpha = 0.5, color = self.colors[i])
                
        plt.scatter(center[:,0], center[:,1], s = 80, color = 'black')
        
        plt.legend(self.based_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=1)
        plt.savefig(filename+'.png', bbox_inches='tight')
        plt.close()

        