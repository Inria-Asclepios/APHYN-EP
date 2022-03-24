from tqdm import tqdm
import numpy as np
import os
import glob
import scipy.io

import torch.utils.data as data
from abc import ABC, abstractmethod

def make_dataset(dir, max_dataset_size=float("inf"), extension = '.npy'):
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

        self.paths_ = sorted(make_dataset(dataroot, max_dataset_size=int(opt.max_dataset_size)))  
        self.len = len(self.paths_) 
        self.extended = opt.data_ext

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains V, stim_points
        """
        item_path = self.paths_[index % self.len]  # make sure index is within then range
        
        np_dict = np.load(item_path, allow_pickle=True)
        
        return np_dict.item()

    def __len__(self):
        """Return the total number of images in the dataset.
        """
        return self.len
