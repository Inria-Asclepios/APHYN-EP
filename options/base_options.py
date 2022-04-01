import argparse
import os
from util import utils

import torch

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        ## basic parameters
        parser.add_argument('--name', required=True, type=str, default='experiment_name', help='name of the experiment. It decides where to store image results and models')
        parser.add_argument('--dataroot', type=str, default=None, help='path to images (should have subfolders train, valid, test)')
        parser.add_argument('--result_dir', type=str, default='./results', help='path to the folder to store image results and models')
        
        ## dataset parameters
        parser.add_argument('--t_pred', type=int, default=0, help='number of starting frame for dynamics prediction')
        parser.add_argument('--domain_size', type=int, default=24, help='size of domain for data generation')

        
        ## model parameters
        parser.add_argument('--intgr_method', type=str, default='euler', help='method used in the integrating scheme [rk4 | euler]')
        parser.add_argument('--dt_int_step', type=float, default=0.1, help='dt step in integrating scheme')
        ### physical model parameters
        parser.add_argument('--disable_phys', action='store_true', help='do not use physical model')
        parser.add_argument('--dx_step', type=float, default=1.0, help='dx step in physical model')
        parser.add_argument('--dt_step', type=float, default=0.05, help='dt step in physical model')
        parser.add_argument('--estim_param_names', type=str, default='', help='estimated physical model paramers names (need to be separated with ",") [d, t_stim, t_in, t_out, t_open, t_close, v_gate]')
        ### data-driven model parameters
        parser.add_argument('--disable_residual', action='store_true', help='do not use data-driven model')
        parser.add_argument('--in_ch', type=int, default=1, help='number of input channels in data-driven model')
        parser.add_argument('--n_fltr_res', type=int, default=8, help='number of filters in the last layer in ResNet')
        parser.add_argument('--n_blocks_res', type=int, default=3, help='number of ResNet-blocks in ResNet')
        parser.add_argument('--n_downsampl_res', type=int, default=1, help='number of downsampling in ResNet')
        parser.add_argument('--padding_type_res', type=str, default='reflect', help='the name of padding layer in conv layers: [reflect | replicate | zero]')
#         parser.add_argument('--norm_type', type=str, default='batch', help='chooses which norm to use in resnet model layers. [batch | instance | none]')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [result_dir/name] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.result_dir, opt.name)
        utils.mkdirs(expr_dir)
        
        if opt.continue_train:
            add = '_NEW'
        else:
            add = ''  
        file_name = os.path.join(expr_dir, '{}_opt{}.txt'.format(opt.phase, add))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            

    def parse(self):
        """Parse our options, create result directory customized suffix for already existing result directories"""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # process opt.suffix
        expr_dir = os.path.join(opt.result_dir, opt.name)
        if os.path.exists(expr_dir):
            if not opt.continue_train:
                suffix = '_NEW'
                opt.name = opt.name + suffix

        self.print_options(opt)

        self.opt = opt
        return self.opt

