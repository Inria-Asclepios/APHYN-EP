from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from copy import copy, deepcopy
import os

import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler

from util.utils import plot_imgs_result


plt.rcParams.update({'figure.max_open_warning': 0})

class Training():  
    """ Some class description """
    def __init__(self, opt, data, model, optimizer, loss_func,
                 loss_train = [], loss_train_norm = [],
                 loss_valid = [], loss_valid_norm = []
                ):

        self.opt = opt

        self.input_data = data
#         self.dataset_label = data_split
        self.forecaster = model
    
        if model.derivative_estimator.phy != None:
            self.estim_param_names = model.derivative_estimator.phy.param_names 
        else:
            self.estim_param_names = []
        
        self.optimizer = optimizer
        self.scheduler_for_lr = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_rule_lr)
        
        self.loss_fn = loss_func
        
        if len(loss_train)!=len(loss_valid):
            raise ValueError('List lengths for training and validation losses are not equal')
        
        self.loss_train = loss_train
        self.loss_train_norm = loss_train_norm
        self.loss_valid = loss_valid
        self.loss_valid_norm = loss_valid_norm

    def run(self,
            create_results=False, 
            test_set_ = None
           ):
        
        try:
            device = self.opt.device
            start_epoch = len(self.loss_train)
            nb_epochs = self.opt.n_epochs

            print('\n\nStart with the {}-th epoch\n\n'.format(start_epoch + 1))
            
            dt_ = self.opt.dt_int_step
            T_pred = self.opt.t_pred
            T_total = self.opt.start_horizon
            
            adapt_horizon_step = self.opt.adapt_horizon_step
            stop_horizon = self.opt.stop_horizon
            adapt_horizont = True
            if self.opt.disable_adapt_hor:
                adapt_horizont = False

            estim_param_names = self.estim_param_names
            
            tany = self.opt.t_any 
            print('t_any : ', tany)
            
            lambd_ = self.opt.init_lmbd_loss 
            tau_2 = self.opt.tau_loss 
            adapatative = True 
            if self.opt.disable_adapt_loss:
                adapatative = False
            
            self.explode_ = False

            bar_epoch = tqdm(range(start_epoch + 1, nb_epochs + 1))
            for ep in bar_epoch:
                forecaster_old_dict = deepcopy(self.forecaster.state_dict())
                
                if adapt_horizont:
                    T_total_adapt = min(T_total * (int(ep/adapt_horizon_step)+1), stop_horizon)
                else:
                    T_total_adapt = T_total 
                print('T_total_adapt : ', T_total_adapt)

                for i, data in enumerate(self.input_data):
                    dataset_label_ = data.name
                    loss_list = []
                    norm_list = []
                   
                    param_list = {}

                    for param in estim_param_names:
                        param_list[param] = []

                    if dataset_label_=='train':
                        self.forecaster.train()
                    else:
                        self.forecaster.eval()

                    for bi, batch in enumerate(data):

                        if tany:
                            T_pred = batch['time']
                            T_total_adapt += T_pred

                        t   = torch.from_numpy(np.arange(T_pred, T_total_adapt,dt_)).float().to(device,dtype=torch.float)  # t
                        v   = batch['V'].to(device, dtype=torch.float)[:,:,0:]  # batch_size,ext,t,x,y
                        v   = v.view(-1, 1, *v.shape[2:])

                        stim_points = batch['stim_point'].to(device, dtype=torch.int) # batch_size,x,y
                        stim_points = stim_points.view(-1, *stim_points.shape[2:])

                        vT = v[:,:, T_pred]
                        target = v[:,:, T_pred:T_total_adapt].permute(2, 0, 1, 3, 4) # t,batch_size,ext,x,y

                        prediction, param_pred = self.forecaster(yT=vT,coord_stim=stim_points, t=t)

                        step_ = 1/dt_
                        ind_ = np.arange(0,prediction.size()[0],step_, dtype=np.int)
                        
                        if dataset_label_=='train': 
                            imgs = [target[:,0,0,], prediction[ind_, 0,0,]]

                        for param in estim_param_names:
                            param_list[param].append(param_pred[param].item())

                        loss_pred = self.loss_fn(prediction[ind_], target)
                        loss_list.append(loss_pred.item())

                        loss = lambd_ * loss_pred

                        if ((not self.forecaster.derivative_estimator.enables_residual) or
                            (self.forecaster.derivative_estimator.res==None) or 
                            (self.forecaster.derivative_estimator.phy==None)):
                            norm = 0.
                            norm_list.append(norm)
                        else: 
                            ## norm for ResNet
                            seq = v[:,:, T_pred] 
                            ele = self.forecaster.derivative_estimator.res(seq)
                            batch_v = v.size(0)
                            ele_vec =  ele.contiguous().view(batch_v, -1)
                            norm = ele.pow(2).mean() 

                            norm_list.append(norm.item())
                            loss += norm

                        if np.isnan(np.mean(norm_list)) or (np.mean(norm_list)> 100) or np.isnan(np.mean(loss_list)):
                            self.explode_ = True
                            self.forecaster = forecaster_old_dict
                            losses = [0]*4
                            return self.forecaster, losses, param_list

                        if dataset_label_=='train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    if dataset_label_=='train':
                        self.loss_train.append(np.mean(loss_list)) 
                        self.loss_train_norm.append(np.mean(norm_list))

                        self.scheduler_for_lr.step(ep)

                        ## for message
                        loss_value = self.loss_train[-1]
                        loss_n_value = self.loss_train_norm[-1]
                    elif dataset_label_=='valid':
                        self.loss_valid.append(np.mean(loss_list))
                        self.loss_valid_norm.append(np.mean(norm_list))
                        ## for message
                        loss_value = self.loss_valid[-1]
                        loss_n_value = self.loss_valid_norm[-1]

                    message_ = 'Epoch {} av_{}_loss : {:.5f}'.format(ep,dataset_label_,loss_value)   
                    for param in estim_param_names:
                        message_ += ',\t avg {} : {:.5f}'.format(param, np.mean(param_list[param]))
                    message_ += ',\t norm : {:.5f}'.format(loss_n_value)


                    print(message_)

                if self.explode_: 
                    print('\n\n\n EXPLODE')
                    break

                if adapatative:
                    lambd_ = lambd_ + tau_2 * self.loss_train[-1]
                    
                #### test + imgs
                if create_results:
                    if((ep%self.opt.save_epoch_freq)==0):
                        self.result_maker(imgs, ep, test_set_img = test_set_)
            
            losses = [self.loss_train, self.loss_train_norm,
                      self.loss_valid, self.loss_valid_norm]
            
            return self.forecaster, losses, param_list
           
        except KeyboardInterrupt: 
            
            print('\n\nLast epoch is {}\n\n'.format(ep))
            loss_length = min(len(self.loss_valid), len(self.loss_valid_norm))
            
            self.loss_train = self.loss_train[:loss_length]
            self.loss_train_norm = self.loss_train_norm[:loss_length]
            self.loss_valid = self.loss_valid[:loss_length]
            self.loss_valid_norm = self.loss_valid_norm[:loss_length]
            
            self.forecaster = forecaster_old_dict
            
            losses = [self.loss_train, self.loss_train_norm,
                      self.loss_valid, self.loss_valid_norm]
            
            return self.forecaster, losses, param_list
            
            
    def result_maker(self, imgs, epoch, test_set_img = None):
        
        path = os.path.join(self.opt.result_dir, self.opt.name)
        
        ## save model
        path_model = os.path.join(path,'model')
        if not os.path.exists(path_model):
            os.mkdir(path_model)
        torch.save(self.forecaster.state_dict(), 
                   os.path.join(path_model,'forecaster_ep_{}'.format(epoch)))
        
        path_img = os.path.join(path,'imgs')
        if not os.path.exists(path_img):
            os.mkdir(path_img)
            
        save_title = os.path.join(path_img, 'result_epoch_{}'.format(epoch))
        save_title_V = os.path.join(path_img, 'result_V_epoch_{}'.format(epoch))
        
        imgs_to_display = {'img': [], 'title': [], 'colorbar': []}
        self._result_maker(imgs, imgs_to_display, epoch, phase_='train')

        ## test img calculations
        t_start_ = 0 
        t_stop_ = 30
        dt_ = self.opt.dt_int_step 
        device = self.opt.device
        
        self.forecaster.eval()

        for bi, batch in enumerate(test_set_img):
            t   = torch.from_numpy(np.arange(t_start_, t_stop_,dt_)).float().to(device,dtype=torch.float)  # t
            v   = batch['V'][:,[0]].to(device, dtype=torch.float)[:,:,]  # batch_size,ext,t,x,y
            v   = v.view(-1, 1, *v.shape[2:])
            
            stim_points = batch['stim_point'][:,[0]].to(device, dtype=torch.int) # batch_size,x,y
            stim_points = stim_points.view(-1, *stim_points.shape[2:])

            vT = v[:,:, t_start_]
            target_t = v[:,:, t_start_:t_stop_].permute(2, 0, 1, 3, 4) # t,batch_size,ext,x,y

            prediction_t, param_pred_t = self.forecaster(yT=vT,coord_stim=stim_points, t=t)

        step_ = 1/dt_
        ind_ = np.arange(0,prediction_t.size()[0],step_, dtype=np.int)
        
        fig_title_ = ''
        for param_ in self.estim_param_names:
            fig_title_ +='{}={:.5f}, '.format(param_, param_pred_t[param_])
        
        ## save test V at fixed points
        point = [10,10]
        plt.figure(figsize=(10,10))
        plt.plot(target_t[:,0,0,point[0],point[1]].data.cpu().numpy().tolist(),label = 'target{}'.format(point))
        plt.plot(prediction_t[ind_,0,0,point[0],point[1]].data.cpu().numpy().tolist(),label = 'pred{}'.format(point))
        point = [5,5]
        plt.plot(target_t[:,0,0,point[0],point[1]].data.cpu().numpy().tolist(),label = 'target{}'.format(point))
        plt.plot(prediction_t[ind_,0,0,point[0],point[1]].data.cpu().numpy().tolist(),label = 'pred{}'.format(point))
        plt.legend()
        plt.title(fig_title_)
        plt.savefig(save_title_V)
        plt.show()
        
        horizont__ = 10 
        
        self._result_maker([target_t[:horizont__,0,0,], prediction_t[ind_[:horizont__],0,0,]], 
                           imgs_to_display, epoch, phase_='test')

        ## save train/test results
        plot_imgs_result(imgs_to_display['img'],
                 list_of_titles = imgs_to_display['title'], 
                 list_of_plot_colorbar = imgs_to_display['colorbar'],
                 figsize = (20,20), save = True, save_title = save_title)

        ## save loss
        plt.figure(figsize=(10,10))
        plt.plot(self.loss_train, label='Pred loss')
        plt.plot(self.loss_train_norm, label='Norm loss')
        plt.plot(self.loss_valid, label='Pred val loss')
        plt.plot(self.loss_valid_norm, label='Norm val loss')

        plt.legend()
        plt.yscale('log')
        plt.savefig(os.path.join(path,'loss'))
        
        f = open(os.path.join(path,'loss.txt'), 'a')
        for inv_ep in range(self.opt.save_epoch_freq,0,-1):
            f.write(str(epoch-inv_ep) + '\t')
            f.write(str(self.loss_train[-inv_ep]) + '\t')
            f.write(str(self.loss_train_norm[-inv_ep]) + '\t')
            f.write(str(self.loss_valid[-inv_ep]) + '\t')
            f.write(str(self.loss_valid_norm[-inv_ep]) + '\n')
        f.close()

        plt.show()

    def _result_maker(self, imgs, imgs_to_display,
                      epoch, phase_='train'):

        img_true = imgs[0].data.cpu().numpy()
        img_fake = imgs[1].data.cpu().numpy()
        
        imgs_to_display['img'].append(img_true)
        imgs_to_display['title'].append("V - {} ground truth, epoch {} : ".format(phase_,epoch))
        imgs_to_display['colorbar'].append(True)

        imgs_to_display['img'].append(img_fake)
        imgs_to_display['title'].append("V - {} generated, epoch {} : ".format(phase_,epoch))
        imgs_to_display['colorbar'].append(True)

        imgs_to_display['img'].append(np.abs(img_true-img_fake))
        imgs_to_display['title'].append("V - {} absolute error, epoch {}, loss {:.3f} : ".format(phase_,
                                                                                                 epoch, 
                                                                                                 self.loss_train[-1]))
        imgs_to_display['colorbar'].append(True)

            
    def lambda_rule_lr(self, epoch):
        '''
        n_epochs_wout_decay - number of epochs with the initial learning rate
        total_nb_epochs - total number of epochs
        '''
        lr_0 = 1
        n_epochs_wout_decay = self.opt.start_ep_decay
        total_nb_epochs = self.opt.n_epochs


        x = max(0, epoch - n_epochs_wout_decay)
        lr_l = lr_0 - x*lr_0/float(total_nb_epochs-n_epochs_wout_decay+1)
        return lr_l

