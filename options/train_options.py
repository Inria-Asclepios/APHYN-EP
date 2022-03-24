from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        ## dataset parameters
        ### for data generation
        parser.add_argument('--data_ext', action='store_true', help='data extending (on on 3 remaining quarters of the cardiac slab)')
        parser.add_argument('--t_any', action='store_true', help='data extending by time')
        parser.add_argument('--t_len', type=int, default=10, help='number of frames per data sample (for train and valid)')
        ### for training
        parser.add_argument('--max_dataset_size', type=float, default=1e+8, help='maximum size of input dataset (for train and valid)')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        
        parser.add_argument('--disable_adapt_hor', action='store_true', help='disable adaptive horizons')
        parser.add_argument('--start_horizon', type=int, default=2, help='total (if "disable_adapt_hor" is ON) or starting number of frames for training forecasting horizon')
        parser.add_argument('--stop_horizon', type=int, default=6, help='terminating number of frames for training forecasting horizon')
        parser.add_argument('--adapt_horizon_step', type=int, default=4, help='step to change the number of frames in adaptive horizon strategy')
        
        
        ## training parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--n_epochs', type=int, default=100, help='total number of epochs')
        parser.add_argument('--lr_init', type=float, default=1e-3, help='initial learning rate for adam')
        parser.add_argument('--start_ep_decay', type=int, default=20, help='starting epoch to linearly decay learning rate to zero')
        parser.add_argument('--loss_reduction', type=str, default='mean', help='specifies the reduction to apply to the loss output: [mean | sum]')
        parser.add_argument('--disable_adapt_loss', action='store_true', help='disable adaptive loss calculations')
        parser.add_argument('--init_lmbd_loss', type=float, default=1., help='initial lambda for adaptative loss calculations')
        parser.add_argument('--tau_loss', type=float, default=1e+3, help='tau for adaptative loss calculations')
        
        
        ## network saving and loading parameters
        parser.add_argument('--no_results', action='store_true', help='do not create and save trainings resilts')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving results at the end of some epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load? if load_iter > 0, the code will load pre-trained models from load_iters epoch; otherwise, the code will load last pre-trained models')
        
        self.isTrain = True
        return parser
