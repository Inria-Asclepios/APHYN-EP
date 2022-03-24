#!/bin/env python3

# import sys
# sys.path.append('../')

from matplotlib import pyplot as plt
from tqdm import tqdm
import os

import numpy as np

from util.dataset_for_generation import DataGenerator

from util.utils import mkdir

import torch

from options.train_options import TrainOptions

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    
    folder = opt.name
    
    mkdir(folder)
    
    time_frames = opt.t_len # max horizont per data sample
    
    if opt.t_any:
        t_any_ = True
    else:
        t_any_ = False

    for phase in ['train', 'valid', 'test']:

        dataset_path = os.path.join(folder,phase)
        mkdir(dataset_path)

        dataset = DataGenerator(opt,phase)

        print('Dataset {} has {} len'.format(phase, len(dataset)))

        with tqdm(total=len(dataset)) as pbar:
            i = 1
            for frame_nb in range(len(dataset)):
                frame_set = dataset[frame_nb]

                v_dim = frame_set['V'].shape[0]

                for j in range(v_dim):
                        file_name = os.path.join(dataset_path, 'sample_{}.npy'.format(i))

                        if (phase=='train')or(phase=='valid'):
                            if t_any_:
                                signif_frames = frame_set['V'].shape[1] - time_frames 
                                data_all_dyn = np.array([])
                                for tm in np.arange(0, signif_frames, 5):
                                    frames_to_add = frame_set['V'][j:(j+1), np.arange(tm, tm+time_frames, 1)]

                                    trash_index_ = np.max(np.std(frames_to_add[:,1:], axis=1), axis=(1,2))>0.02
                                    if trash_index_:
                                        file_name_ = file_name[:-4]+'_{}.npy'.format(tm)
                                        frame_set_ = {'V': frames_to_add, 
                                                      'stim_point': frame_set['stim_point'][[j]], 
                                                      'time' : tm}
                                        np.save(file_name_, frame_set_)

                            else:
                                frame_set_ = {'V': frame_set['V'][[j], :30], 'stim_point': frame_set['stim_point'][[j]]} 
                        else:
                            frame_set_ = {'V': frame_set['V'][[j]], 'stim_point': frame_set['stim_point'][[j]]}

                        np.save(file_name, frame_set_)
                        i+=1

                pbar.update(1)