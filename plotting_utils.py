import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot_with_boxes(t1,x1,t2,x2,n_boxes,pos_boxes, desc = '', noise = None, c = 'b', fs = 86, lw = 13, final_t = max):

    if noise is not None:
        desc += ', ' + str(int(100*noise)) + "% noise"

    if t2 is None:
        bot = np.min(x1)
        top = np.max(x1)
    else:
        bot = min(np.min(x1),np.min(x2))
        top = max(np.max(x1),np.max(x2))
    
    bot = np.log10(0.98 * 10 ** bot)
    scale_alpha = 0.99

    fig, ax = plt.subplots(figsize=(64, 48), layout='constrained')
    
    lab = "bi-level"
    #if "error" in desc:
    #    plt.ylabel("rel. error", fontsize = fs)
    #elif "res" in desc:
    #    plt.ylabel("rel. res", fontsize = fs)

    plt.semilogy(t1,x1,c = c, linewidth = lw, label = lab)
    if t2 is not None:
        plt.semilogy(t2,x2,'k--', linewidth = lw, label = "direct Landweber")
    
    for i, alpha in enumerate(range(n_boxes)):
        extra = 0
        if i == n_boxes - 1:
            if t2 is None:
                extra = np.max(t1) - pos_boxes[i+1]
            else:
                extra = max(np.max(t1),np.max(t2)) - pos_boxes[i+1]
        alpha = scale_alpha * (alpha+1)/n_boxes
        ax.add_patch(Rectangle((pos_boxes[i], bot), pos_boxes[i+1]-pos_boxes[i]+extra, top-bot, alpha=alpha, zorder=0))
        #ax.text((pos_boxes[i+1]+pos_boxes[i])/2, top + 0.01*np.abs(top), "Rf. " + str(i), ha='center', fontsize = fs)
    if t2 is None:
        plt.xlim(0,np.max(t1))
    else:
        plt.xlim(0,final_t(np.max(t1),np.max(t2)))
        
    #ax.set_xscale('function', functions=(lambda x: (x+1)**(1/2), lambda x: x**2-1))
    
    plt.xticks(fontsize=fs)
    ax.tick_params(axis = 'y', which = 'both', labelsize = fs)
    
    plt.title(desc,fontsize=2*fs)
    plt.xlabel("seconds", fontsize = fs)
    if "error" in desc:
        plt.ylabel("relative error", fontsize = fs)
    elif "Res" in desc:
        plt.ylabel("relative residual", fontsize = fs)
    plt.legend(loc = "right", fontsize = fs)
    plt.savefig(desc.replace(".","_") + ".png")
    
    #return fig, ax
