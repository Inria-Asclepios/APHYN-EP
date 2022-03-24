import torch.utils.data
import numpy as np

from .dataset import DataGenerator

class DataLoader():
    """Dataset class that performs multi-threaded data loading"""
    
    def __init__(self, opt, phase):
        self.opt = opt
        self.dataset = DataGenerator(opt, phase)
        self.name = phase
        sampler = torch.utils.data.sampler.SubsetRandomSampler(np.arange(self.dataset.len))
        if phase=='test':
            sampler = None

        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      sampler=sampler,
                                                      batch_size=opt.batch_size,
                                                      pin_memory=True,
                                                      drop_last=False #True
                                                     )
        
        
        
    def load_data(self):
        return self
    
    def __len__(self):
        """Return the number of data in the dataset"""
        return min(self.dataset.len, self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data