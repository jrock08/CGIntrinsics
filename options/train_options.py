from .base_options import BaseOptions
import os
from util import util

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--no_flip'  , action='store_true', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--IIW', action='store_true')
        self.parser.add_argument('--SAW', action='store_true')
        self.parser.add_argument('--pretrained_cgi', action='store_true', help='drastically changes how to think about the training, instead of all training at once, assumption is you have a fixed image-to-image model, and you just need to train some ancilary stuff that does IIW.  This removes a bunch of the losses related to smoothness and turns off CGI during training.')
        self.parser.add_argument('--hot_start', action='store_true')
        self.parser.add_argument('--progressive_iiw_weight', action='store_true')
        self.parser.add_argument('--num_train_epochs', type=int, default=50)
        self.parser.add_argument('--IIW_pair_clamp', type=int, default=1000, help='most of the images have a few hundred pairs, but a couple have 10ks, if you batch them together the gpus will oom, this avoids that')
        # NOT-IMPLEMENTED self.parser.add_argument('--preprocessing', type=str, default='resize_and_crop', help='resizing/cropping strategy')
        self.isTrain = True

    def parse(self):
        super(TrainOptions, self).parse()

        args = vars(self.opt)

        # save to the disk
        expr_dir =  os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
