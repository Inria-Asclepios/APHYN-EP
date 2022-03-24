import os
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({'figure.max_open_warning': 0})

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def plot_imgs(list_of_img, title = '', nb_col = 2, figsize = (15,15), 
              plot_colorbar = False, save = False, save_title = ' ', 
              path = ''):
    
    len_list_of_img = len(list_of_img)
    
    if (len_list_of_img*1. % nb_col):
        nb_row = int((len_list_of_img*1./nb_col)+1)
    else :
        nb_row = int(len_list_of_img*1./nb_col)
    
    fig, axs = plt.subplots(nrows=nb_row, ncols=nb_col, 
                            figsize=figsize)
    axs_n = np.reshape(axs, (nb_row, nb_col)) ### to avoid 1 shape error
    
    fig.suptitle(title, y=1.05, fontsize=16)
    
    for i in range(len_list_of_img): 
        j,k = (int(i / nb_col), int(i % nb_col))
        im = axs_n[j, k].imshow(list_of_img[i])
        axs_n[j, k].set_axis_off()
        
        if plot_colorbar:
            plt.colorbar(im,ax=axs_n[j, k])
                  
    if save :
        plt.savefig(path+save_title)
        
    plt.close()
    
def plot_imgs_result(list_of_list_imgs, list_of_titles = None, list_of_plot_colorbar = None, 
                     figsize = (20, 20), save = False, save_title = ' ', path = ''):
    
    nb_of_lists = len(list_of_list_imgs)
    
    fig = plt.figure(figsize=figsize)
    outer_grid = fig.add_gridspec(nb_of_lists, 1, wspace=0, hspace=0.5)
    
    for i in range(nb_of_lists):
    
        nb_col = len(list_of_list_imgs[i])
        nb_row = 1 
            
        inner_grid = outer_grid[i, 0].subgridspec(nb_row, nb_col, wspace=0.05, hspace=0.05)
        axs = inner_grid.subplots()  # Create all subplots for the inner grid.
        axs_n = np.reshape(axs, (nb_row, nb_col)) ### to avoid 1 shape error
        
        for (j, k), ax in np.ndenumerate(axs_n):
            im = ax.imshow(list_of_list_imgs[i][j*nb_row+k])
            ax.set_axis_off()
            ax.set_title(list_of_titles[i]).set_visible(ax.is_first_col())
            
            if list_of_plot_colorbar[i]:
                plt.colorbar(im,ax=ax)
    
    if save :
        plt.savefig(path+save_title)
        
    plt.close()
