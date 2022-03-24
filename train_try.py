#!/bin/env python3

# import sys
# sys.path.append('../')

import numpy as np
from matplotlib import pyplot as plt
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
from util.networks_my import *
from util.dataloader import DataLoader
import util.utils

from util.resnet import Resnet_2D

from options.train_options import TrainOptions

plt.rcParams.update({'figure.max_open_warning': 0})

def get_param_names(string):
    return prim.split(' ')

if __name__ == '__main__':
    
    opt = TrainOptions().parse()   # get training options
    
    dataset_train = DataLoader(opt, 'train').load_data()
    dataset_valid = DataLoader(opt, 'valid').load_data()
    dataset_test = DataLoader(opt, 'test').load_data()
    
    estim_param_names = opt.estim_param_names.split(' ')
    if len(estim_param_names)==1:
        if estim_param_names[0] == '':
            estim_param_names = []
    
    ## Physical model F_phy, Laplacian2 to use a determined physical model
    if opt.disable_phys:
        physical_model = None
    else:
        physical_model = Laplacian_2(dx_step = opt.dx_step, estim_param_names = estim_param_names, 
                               n_domain= opt.domain_size, device = opt.device)

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

    loss_fn = nn.MSELoss(reduction=opt.loss_reduction)
    
    optimizer = optim.Adam(forecaster.parameters(), lr=opt.lr_init)

    scheduler_for_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule_lr)
    
    forecaster.derivative_estimator.phy._dt = opt.dt_step
    forecaster.derivative_estimator.phy.ext_dt = opt.dt_int_step
    
    for param in physical_model.parameters():
        print(param)
        
    device = opt.device
    epochs = opt.n_epochs
    
    T_pred = opt.t_pred
    
    T_total = opt.start_horizon
    adapt_horizont = True
    if opt.disable_adapt_hor:
        adapt_horizont = False
    
    save_results = True
    if opt.no_results:
        save_results = False

    tany = opt.t_any #////////////////////////////////////////////////////////////
    print('t_any : ', tany)

    dt_ = opt.dt_int_step
    lambd_ = opt.init_lmbd_loss 
    tau_2 = opt.tau_loss 
    adapatative = True 
    if opt.disable_adapt_loss:
        adapatative = False
    
    loss_train = []
    loss_train_norm = []
    loss_valid = []
    loss_valid_norm = []

    start_epoch = 0 #////////////////////////////////////////////////////////////
    dataset = [dataset_train, dataset_valid]
    split = ['train', 'valid']

    explode_ = False

    bar_epoch = tqdm(range(start_epoch + 1, epochs + 1)) #epochs
    for ep in bar_epoch:

        #////////////////////////////////////////////////////////////
        torch.save(forecaster.state_dict(), os.path.join(os.path.join(opt.result_dir, opt.name), 'forecaster'))

        forecaster_old_dict = deepcopy(forecaster.state_dict())
        
        if adapt_horizont:
            T_total_adapt = min(T_total * (int(ep/opt.adapt_horizon_step)+1), opt.stop_horizon)
        else:
            T_total_adapt = T_total 
        print('T_total_adapt : ', T_total_adapt)

        for i, data in enumerate(dataset):
            loss_list = []
            norm_list = []

            param_list = {}

            for param in estim_param_names:
                param_list[param] = []

            if split[i]=='train':
                forecaster.train()
            else:
                forecaster.eval()

            for bi, batch in enumerate(data):

                if tany:
                    T_pred = batch['time']
                    T_total_adapt += T_pred

                t   = torch.from_numpy(np.arange(T_pred, T_total_adapt,dt_)).float().to(device,dtype=torch.float)  # t
                v   = batch['V'].to(device, dtype=torch.float)[:,:,1:]  # batch_size,ext,t,x,y
                v   = v.view(-1, 1, *v.shape[2:])

                stim_points = batch['stim_point'].to(device, dtype=torch.int) # batch_size,x,y
                stim_points = stim_points.view(-1, *stim_points.shape[2:])

                vT = v[:,:, T_pred]
                target = v[:,:, T_pred:T_total_adapt].permute(2, 0, 1, 3, 4) # t,batch_size,ext,x,y

                prediction, param_pred = forecaster(yT=vT,coord_stim=stim_points, t=t)

                step_ = 1/dt_
                ind_ = np.arange(0,prediction.size()[0],step_, dtype=np.int)

                for param in estim_param_names:
                    param_list[param].append(param_pred[param].item())

                loss_pred = loss_fn(prediction[ind_], target)
                loss_list.append(loss_pred.item())
                
                loss = lambd_ * loss_pred

                if (not enables_residual)or(residual_model==None):
                    norm = 0.
                    norm_list.append(norm)
                else: 
                    ## norm for my ResNet
                    seq = v[:,:, T_pred] 
                    ele = residual_model(seq)
                    batch_v = v.size(0)
                    ele_vec =  ele.contiguous().view(batch_v, -1)
                    norm = ele.pow(2).mean() 

                    norm_list.append(norm.item())
                    loss += norm

                if np.isnan(np.mean(norm_list)) or (np.mean(norm_list)> 100) or np.isnan(np.mean(loss_list)):
                    explode_ = True
                    break

                if split[i]=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if split[i]=='train':
                loss_train.append(np.mean(loss_list)) 
                loss_train_norm.append(np.mean(norm_list))

                scheduler_for_lr.step(ep)

                ## for message
                loss_value = loss_train[-1]
                loss_n_value = loss_train_norm[-1]
            elif split[i]=='valid':
                loss_valid.append(np.mean(loss_list))
                loss_valid_norm.append(np.mean(norm_list))
                ## for message
                loss_value = loss_valid[-1]
                loss_n_value = loss_valid_norm[-1]

            message_ = 'Epoch {} av_{}_loss : {:.5f}'.format(ep,split[i],loss_value)   
            for param in estim_param_names:
                message_ += ',\t avg {} : {:.5f}'.format(param, np.mean(param_list[param]))
            message_ += ',\t norm : {:.5f}'.format(loss_n_value)

        #     bar_epoch.set_postfix_str(message_)

            print(message_)

        if explode_:
            print('\n\n\n EXPLODE')
            break

        if adapatative:
            lambd_ = lambd_ + tau_2 * loss_valid[-1] # loss_train[-1]

    if explode_:
        forecaster.load_state_dict(deepcopy(forecaster_old_dict))

    torch.save(forecaster.state_dict(), os.path.join(os.path.join(opt.result_dir, opt.name), 'forecaster'))
