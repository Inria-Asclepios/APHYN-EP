from tqdm import tqdm
import numpy as np
import os
import glob
import scipy.io

import torch.utils.data as data
from abc import ABC, abstractmethod

def make_dataset(dir, max_dataset_size=float("inf"), extension = '.mat'):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    img_paths = glob.glob(dir+'/*{}'.format(extension))
    return img_paths[:min(max_dataset_size, len(img_paths))]


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        
    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0
    
    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
    
class DataGenerator(BaseDataset):
    """
    This dataset class can load dataset.
    """

    def __init__(self, opt, phase):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        dataroot = os.path.join(opt.dataroot, phase)

        self.paths_ = sorted(make_dataset(dataroot, opt.max_dataset_size))  
        self.len = len(self.paths_) 
        if opt.data_ext:
            self.extended = True
        else:
            self.extended = False

    def __getitem__(self, index, pot_min = -90, pot_max = 40, n_domain = 24):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains V, stim_points
        """
        item_path = self.paths_[index % self.len]  # make sure index is within then range
        
        mat = scipy.io.loadmat(item_path)
        mat['potential_norm'] = (mat['potential']-pot_min)/(pot_max-pot_min)
            
        potential_ = mat['potential_norm'][:,:,0].transpose(2, 0, 1)[1:,:n_domain,:n_domain] ## not include V = 0 first frame
        axis_expand = 0
        coord_x, coord_y, _ = mat['stimCoords'][0]
        
        if self.extended:
            potential_ext = [potential_]
            potential_ext.append(np.flip(potential_, axis=2))
            potential_ext.append(np.flip(potential_, axis=1))
            potential_ext.append(np.flip(potential_ext[-1], axis=2))
            V_res = np.array(potential_ext)
              
            coord_ext = [[coord_x, coord_y]]
            coord_ext.append([coord_x, n_domain-coord_y])
            coord_ext.append([n_domain-coord_x, coord_y])
            coord_ext.append([n_domain-coord_x, n_domain-coord_y])
            coord_res = np.array(coord_ext)
            
        else:
            V_res = np.expand_dims(potential_,axis=0)
            coord_res = np.array([coord_x, coord_y])
        
        return {'V': V_res, 'stim_point':coord_res}

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.len
