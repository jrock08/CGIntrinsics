import argparse
import os
from util import util

class BaseOptions(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        # self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
        # self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--name', type=str, default='test_local', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--sub_name', type=str, default='', help='second name for the experiment, determines the model name')
        # self.parser.add_argument('--align_data', action='store_true',
                                # help='if True, the datasets are loaded from "test" and "train" directories and the data pairs are aligned')
        self.parser.add_argument('--model', type=str, default='pix2pix',
                                 help='chooses which model to use. cycle_gan, one_direction_test, pix2pix, ...')
        # self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.parser.add_argument('--human_judgement_gray', action='store_true', help='treat the physical and human judgement as two different projections of the true RGB reflectance image, rather than the same grayscale image')

        self.parser.add_argument('--human_judgement_model', type=str, default='mlp', help='simple, mlp, or residual, must also set human_judgement_gray')
        self.parser.add_argument('--human_pair_classifier', action='store_true', help='rather than using image ratios, use a small MLP to predict which pixel is brighter')
        self.parser.add_argument('--human_pair_classifier_type', type=str, default='ternary', help='ternary, binary, single_score, single_score_const_thresh')
        self.parser.add_argument('--bilinear_classifier',action='store_true')

        self.parser.add_argument('--output_reflectance_dim', type=int, default=-1)
        self.parser.add_argument('--append_chroma', action='store_true')
        self.parser.add_argument('--num_pyr_levels', type=int, default=1)
        self.parser.add_argument('--iiw_weight', type=float, default=4.0)
        self.parser.add_argument('--detach_iiw_loss', action='store_true')
        self.parser.add_argument('--use_base_IIW', action='store_true')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt
