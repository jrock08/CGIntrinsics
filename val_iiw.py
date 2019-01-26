import time
import torch
import sys
from scipy import misc
import h5py
import numpy as np

from options.test_options import TestOptions
from data.data_loader import CreateDataLoaderCGIntrinsics
from data.data_loader import CreateDataLoaderCGIntrinsicsTest
from data.data_loader import CreateDataLoaderIIW
#from data.data_loader import CreateDataLoaderIIWTest
from data.data_loader import CreateDataLoaderIIWVal
from data.data_loader import CreateDataLoaderRender
from data.data_loader import CreateDataLoaderSAW

from models.models import create_model
import torch
import math

opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


full_root = '/data/jrock/IIW_2019/'

val_list_IIW = full_root + '/CGIntrinsics/IIW/train_val_list/val_list/'
data_loader_val_IIW = CreateDataLoaderIIWVal(full_root, val_list_IIW, 0, 1)
dataset_val_size_IIW = len(data_loader_val_IIW)
dataset_val_IIW = data_loader_val_IIW.load_data()

val_list_CGIntrinsics = full_root + '/CGIntrinsics/intrinsics_final/train_val_list/val_list/'
data_loader_val_CGI = CreateDataLoaderCGIntrinsicsTest(full_root, val_list_CGIntrinsics, 1)
dataset_val_CGIntrinsics = data_loader_val_CGI.load_data()
dataset_val_size_CGIntrinsics = len(data_loader_val_CGI)

model = create_model(opt)
model.switch_to_eval()

epoch = 0
IIW_val_loss = 0
CGI_val_loss = 0
for i, data in enumerate(dataset_val_IIW):
    stacked_img = data['img_1']
    targets = data['target_1']

    model.set_input(stacked_img, targets)
    data_set_name = 'IIW'
    IIW_val_loss += model.val_eval_loss(epoch, data_set_name)
    #IIW_val_loss += model.validate_intrinsics(epoch, data_set_name)
print 'IIW {}'.format(IIW_val_loss)

for i, data in enumerate(dataset_val_CGIntrinsics):
    stacked_img = data['img_1']
    targets = data['target_1']

    model.set_input(stacked_img, targets)
    data_set_name = 'CGIntrinsics'
    CGI_val_loss += model.val_eval_loss(epoch, data_set_name)

print 'CGI {}'.format(CGI_val_loss)

