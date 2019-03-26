from test_iiw import test_iiw

import time
import torch
import numpy as np
from options.test_options import TestOptions
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from models.models import create_model
# from data.data_loader import CreateDataLoader_TEST
from data.data_loader import CreateDataLoaderIIWTest
from data.data_loader import CreateDataLoaderIIW
import torchvision

if __name__ == '__main__':
    import os
    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    model = create_model(opt)

    whdr_thresholds = np.linspace(0,1,21)
    val_outp = test_iiw(model, 'train_val_list/val_list/', thresholds=whdr_thresholds)
    whdr_test = whdr_thresholds[np.argmin(val_outp[0])]

    img_out_dir = '/data/jrock/out_relight_2019/iiw/' + opt.name + '_' + opt.sub_name + '/'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    outp = test_iiw(model, 'test_list/', thresholds=[whdr_test], img_out_dir=img_out_dir)

    with open('val_whdr.txt','a') as f:
        f.write(opt.name + ' ' + opt.sub_name + '\n')
        f.write('best whdr threshold: {}\n'.format(whdr_test))
        f.write('WHDR {}\n'.format(outp[0]))
        f.write('WHDR_EQ {}\n'.format(outp[1]))
        f.write('WHDR_INEQ {}\n'.format(outp[2]))

    print 'best whdr threshold: {}'.format(whdr_test)
    print 'WHDR {}'.format(outp[0])
    print 'WHDR_EQ {}'.format(outp[1])
    print 'WHDR_INEQ {}'.format(outp[2])
    #for WHDR, WHDR_EQ, WHDR_INEQ in outp:
    #    print('WHDR %f'%WHDR)

    #WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'test_list/')
    #WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'train_val_list/val_list/')

    print("We are done")

