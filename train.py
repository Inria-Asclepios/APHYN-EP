#!/bin/env python3

import numpy as np
import glob
import os
from tqdm import tqdm
import scipy.io
from copy import copy, deepcopy

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn
from torch.nn import init
from torch import optim
from torch.utils.data import DataLoader, sampler
from torch.nn import Parameter
from torch.optim import lr_scheduler

from torchdiffeq import odeint as odeint
from util.networks import *
from util.training_template import Training

from util.dataloader import DataLoader
import util.utils

from util.resnet import Resnet_2D

from options.train_options import TrainOptions

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def get_param_names(string):
    return prim.split(' ')

if __name__ == '__main__':
    
    opt = TrainOptions().parse()   # get training options
    
    dataset_train = DataLoader(opt, 'train').load_data()
    dataset_valid = DataLoader(opt, 'valid').load_data()
    dataset_test = DataLoader(opt, 'test').load_data()
    
    dataset = [dataset_train, dataset_valid]
    
    ### add data info in option file
    if opt.continue_train:
            add = '_NEW'
    else:
            add = ''  
    opt_file_name = os.path.join(os.path.join(opt.result_dir, opt.name), 
                                 '{}_opt{}.txt'.format(opt.phase, add))
    message = '\n'
    for data_ in dataset:
        len_ = len(data_)
        message += '\n dataset {} with len : {}'.format(data_.name, len_)
    with open(opt_file_name, 'a') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
    
    ### Model 
    estim_param_names = opt.estim_param_names.split(',')
    if len(estim_param_names)==1:
        if estim_param_names[0] == '':
            estim_param_names = []
    print("\n estim_param_names : ", estim_param_names)
    
    ## Physical model F_phy, Laplacian2 to use a determined physical model
    if opt.disable_phys:
        physical_model = None
        enables_residual = False
    else:
        physical_model = Laplacian_2(dx_step = opt.dx_step, estim_param_names = estim_param_names, 
                               n_domain= opt.domain_size, device = opt.device)
        ## print parameters
        for param in physical_model.parameters():
            print(param)

    ## Residual model F_a
    if opt.disable_residual:
        enables_residual = False
        residual_model = None
    else:
        enables_residual = True
        residual_model = Resnet_2D(input_nc=opt.in_ch, output_nc=opt.in_ch, ngf=opt.n_fltr_res, 
                                   norm_layer=nn.BatchNorm2d, n_blocks=opt.n_blocks_res, 
                                   padding_type=opt.padding_type_res, 
                                   n_downsampling=opt.n_downsampl_res)
        ## Orthogonal weight initialisations for ResNet
        for m in residual_model.modules():
            if m.__class__.__name__.find('Conv') != -1:
                nn.init.orthogonal_(m.weight.data, gain=0.0001)
                print('Orthogonal weight initialisations for ResNet module - ', m.__class__.__name__)


    ## dX/dt = F(X,t) = F_phy + F_a
    derivative_estimator = DerivativeEstimator(physical_model=physical_model, 
                                               residual_model= residual_model, 
                                               enables_residual=enables_residual)

    forecaster = Forecaster(derivative_estimator, method=opt.intgr_method).to(opt.device)
    
    loss_train = []
    loss_train_norm = []
    loss_valid = []
    loss_valid_norm = []
    
    #### Fine-tuning/resume training ####
    if opt.continue_train:
        ### Upload losses
        expr_dir = os.path.join(opt.result_dir, opt.name)
        file_name = os.path.join(expr_dir, 'loss.txt')
        f = open(file_name)
        for line in f:
            ep, loss1, loss2, loss3, loss4 = line.split('\t')
            loss_train.append(float(loss1))
            loss_train_norm.append(float(loss2))
            loss_valid.append(float(loss3))
            loss_valid_norm.append(float(loss4))
        f.close()
            
        if opt.load_iter:
            epoch = opt.load_iter
            ### Re-write loss file
            if opt.load_iter<int(ep)+1:
                os.rename(file_name, file_name[:-4]+'_old.txt')
                f = open(file_name, 'wt')
                for i in range(epoch):
                    f.write(str(i) + '\t')
                    f.write(str(loss_train[i]) + '\t')
                    f.write(str(loss_train_norm[i]) + '\t')
                    f.write(str(loss_valid[i]) + '\t')
                    f.write(str(loss_valid_norm[i]) + '\n')
                f.close()
                
                loss_train = loss_train[:epoch]
                loss_train_norm = loss_train_norm[:epoch]
                loss_valid = loss_valid[:epoch]
                loss_valid_norm = loss_valid_norm[:epoch]
        else:
            epoch = int(ep)+1
        
        ### Upload pre-trained model
        path_model = os.path.join(expr_dir,'model')
        forecaster.load_state_dict(torch.load(os.path.join(path_model,'forecaster_ep_{}'.format(epoch))))
        forecaster.eval()
     
    if physical_model != None:
        forecaster.derivative_estimator.phy._dt = opt.dt_step
        forecaster.derivative_estimator.phy.ext_dt = opt.dt_int_step

    optimizer = optim.Adam(forecaster.parameters(), lr=opt.lr_init)
    loss_fn = nn.MSELoss(reduction=opt.loss_reduction)

    create_results = True
    if opt.no_results:
        create_results = False
        
    train_func = Training(opt, dataset, forecaster, optimizer, loss_fn,
                          loss_train = loss_train,
                          loss_train_norm = loss_train_norm,
                          loss_valid = loss_valid,
                          loss_valid_norm = loss_valid_norm,
                         )
    
    forecaster, lossses, parameters = train_func.run(create_results=create_results, 
                                                     test_set_ = dataset_test
                                                    )
    
    