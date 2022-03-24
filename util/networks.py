import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torchdiffeq import odeint
from torch import optim
import functools
from copy import copy, deepcopy


from torch.nn.utils import spectral_norm as sn

import math
import numbers
from scipy import ndimage


class DerivativeEstimator(nn.Module):

    def __init__(self, physical_model, residual_model, enables_residual):

        # enables_residual == True => F_a, F_phy decomposition : F(X,t)=F_a + F_phy 
        # enables_residual == False => F(X,t)=F_phy

        super().__init__()
        self.enables_residual = enables_residual

        self.phy = physical_model
        self.res = residual_model
        
    def forward(self, t, x):
        if self.phy==None:
            res = self.res(x)
            out = res
        else:
            phy = self.phy(t, x)
            if self.enables_residual:
                res = self.res(x)
                out = phy + res
            else:
                out = phy
        
        return out

    
class Forecaster(nn.Module):
    def __init__(self, derivative_estimator, int_=odeint, method=None):  
        ### method : euler, rk4 etc.
        super().__init__()

        self.derivative_estimator = derivative_estimator
        self.int_ = int_
        self.method = method

    def forward(self, yT, coord_stim, t=None):
        
        phy_mod = self.derivative_estimator.phy
        
        if phy_mod!=None:
            phy_mod.coord_stim = coord_stim
            phy_mod.H = None

        ## solves dy/dt = func(t, y), y(t[0]) = y0, odeint(func, y0, ...)
        forecasts = self.int_(self.derivative_estimator, y0=yT, t=t, method=self.method) 

        ### get physical model parameters
        if phy_mod!=None:
            parameters = phy_mod.param
        else:
            parameters = [0]

        return forecasts[:,:,[0]], parameters
    

### F_phy(V) -> V              
class Laplacian_2(nn.Module):

    def __init__(self, dx_step, estim_param_names, n_domain, device, network=False):

        super().__init__()

        self._dx = dx_step
        self.device = device
        
        self._kernel = nn.Parameter(torch.tensor(
            [
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0],
            ],
        ).float().view(1, 1, 3, 3) / (self._dx * self._dx), requires_grad=False)

        self.n_domain = n_domain
        self.network = network
        
        self.param_names = estim_param_names

        self.param = {}
        self.fc_param = nn.ParameterDict({})
        
        ### Real values 
        self.param['d'] = 0.1*1.
        self.param['t_in'] = 0.1/0.3
        self.param['t_out'] = 1./6.
        self.param['t_open'] = 100./120.
        self.param['t_close'] = 100./150
        self.param['v_gate'] = 0.13
        self.param['t_stim'] = 0.1*0.1
        
        self.coord_stim = None
        self.window = [5]
        
        self.V0 = None 
        self.H0 = None 
        self.H = None
        self._dt = None
        self.ext_dt = None
        

        for name in self.param_names:
            self.param[name] = None
            
            if network:
                pass
            else:
                self.fc_param['fc_'+name] = nn.Parameter(torch.tensor(0.2), requires_grad=True) #-2. 
    
    def calc_H(self, t, t0_=0, x_vh_=None, int_=odeint, method=None):
        t_int = torch.from_numpy(np.arange(t0_, t.cpu()+0.5*self._dt, self._dt)).to(self.device,dtype=torch.float) 
        
        if x_vh_==None:
            y0 = torch.cat([self.V0, self.H0], dim=1)
        else:
            y0 = x_vh_
            
        forecast = int_(self._physical_model, y0=y0, t=t_int, method='euler')
        return forecast[-1,:,[1],]
        
    
    def forward(self, t, x_v):
        V = x_v
        
        if self.network:
            pass
        else:
            for name in self.param_names:
                self.param[name] = self.fc_param['fc_'+name]

        if t == 0:
            self.V0 = deepcopy(V)
            self.H0 = torch.from_numpy(np.ones((V.size()[0], 1, 
                                                self.n_domain, 
                                                self.n_domain))).to(self.device,dtype=torch.float) 
            H = self.H0
        else:
            if self.H == None: 
                self.H0 = torch.from_numpy(np.ones((V.size()[0], 1, 
                                                    self.n_domain, 
                                                    self.n_domain))).to(self.device,dtype=torch.float) 
                
                self.V0 = torch.from_numpy(np.zeros((V.size()[0], 1, 
                                                     self.n_domain, 
                                                     self.n_domain))).to(self.device,dtype=torch.float)
                self.H = self.calc_H(t)
           
                
            H = self.H
    
        x_vh = torch.cat([V, H], dim=1)
        vh_phy = self._physical_model(t, x_vh)
        
        self.H = self.calc_H(t+1.*self.ext_dt, t0_=t.cpu(), x_vh_=x_vh) 
        
        return vh_phy[:,[0]]
    
    def _physical_model (self,t,x_vh):
        
        V = x_vh[:,[0]]
        H = x_vh[:,[1]]
        
        coord_stim = self.coord_stim
        window = self.window
        
        ## F is torch.nn.functional 
        V_ = F.pad(V, pad=(1,1,1,1), mode='replicate')
        
        Delta_v = F.conv2d(V_, self._kernel.to(self.device, dtype=torch.float))
        J_stim_ = self.J_stim(t,self.param['t_stim']*10.,coord_stim, window)
        
        
        output_v = (self.param['d']*10.* Delta_v + \
                    self.param['t_in']*10.*(1.-V)*V.pow(2)*H - \
                    self.param['t_out']*V + J_stim_)
        output_h = ((self.param['t_open']/100.)*self.sign_(self.param['v_gate'], V)*(1-H)-
                    (self.param['t_close']/100.)*self.sign_(V, self.param['v_gate'])*H)
        
        return torch.cat([output_v, output_h], dim=1)
        
    
    def J_stim (self, t, t_stim, coord_stim, window = [5], intensity = [1.], t_0_stim = [0.]):
        
        n_points = coord_stim.size()[0]
        result = torch.zeros((n_points,1,self.n_domain, self.n_domain)).to(self.device, dtype=torch.float)
        
        if (len(window)==1)and(n_points>1):
            window = window*n_points
        
        if (len(intensity)==1)and(n_points>1):
            intensity = intensity*n_points
        
        if (len(t_0_stim)==1)and(n_points>1):
            t_0_stim = t_0_stim*n_points
        
        for i in range(n_points):
            if (t > t_0_stim[i]):
                result[i,:, coord_stim[i,0]:coord_stim[i,0]+window[i],
                       coord_stim[i,1]:coord_stim[i,1]+window[i]] = intensity[i]*self.sign_(t_0_stim[i]+t_stim, t)
            
        return result
    
    def sign_ (self, x1, x2, torch_ = True):
        if torch_:
            return 0.5*(1+torch.sign(x1-x2))
        else :
            return 0.5*(1+np.sign(x1-x2))
        