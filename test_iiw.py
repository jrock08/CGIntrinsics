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

opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

#root = "/home/zl548/phoenix24/"
#full_root = root +'/phoenix/S6/zl548/'

full_root = '/data/jrock/IIW_2019/'
model = create_model(opt)


def test_iiw(model, list_name):
    # print("============================= Validation ============================")
    model.switch_to_eval()

    outp = []
    #thresholds = np.linspace(0,1,15)
    thresholds = [.1]
    total_loss = np.zeros(len(thresholds))
    total_loss_eq = np.zeros(len(thresholds))
    total_loss_ineq = np.zeros(len(thresholds))
    total_count = 0.0
    # for 3 different orientation
    for j in range(0,3):
        # print("============================= Testing EVAL MODE ============================", j)
        test_list_dir = full_root + '/CGIntrinsics/IIW/' + list_name
        print(test_list_dir)
        # True test dataloader
        data_loader_IIW_TEST = CreateDataLoaderIIWTest(full_root, test_list_dir, j)
        dataset_iiw_test = data_loader_IIW_TEST.load_data()

        #data_loader_IIW_TEST = CreateDataLoaderIIW(full_root, test_list_dir, j)
        #dataset_iiw_test = data_loader_IIW_TEST.load_data()

        for i, data in enumerate(dataset_iiw_test):
            stacked_img = data['img_1']
            targets = data['target_1']
            #model.set_input(stacked_img, targets)
            #model.val_eval_loss(0, 'IIW')
            [total_whdr, total_whdr_eq, total_whdr_ineq, count], pred_rgb = model.evlaute_iiw(stacked_img, targets, thresholds)
            torchvision.utils.save_image(pred_rgb, 'out_test_rgb.png')
            torchvision.utils.save_image(stacked_img, 'in_test_rgb.png')
            total_loss += total_whdr
            total_loss_eq += total_whdr_eq
            total_loss_ineq += total_whdr_ineq

            total_count += count
            print("Testing WHDR error ", j, i, total_loss/total_count)

    return total_loss/(total_count), total_loss_eq/total_count, total_loss_ineq/total_count


#print("WE ARE IN TESTING PHASE!!!!")
outp = test_iiw(model, 'train_val_list/val_list/')
with open('val_whdr.txt','a') as f:
    f.write('WHDR {}\n'.format(outp[0]))
print 'WHDR {}'.format(outp[0])
print 'WHDR_EQ {}'.format(outp[1])
print 'WHDR_INEQ {}'.format(outp[2])
#for WHDR, WHDR_EQ, WHDR_INEQ in outp:
#    print('WHDR %f'%WHDR)

#WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'test_list/')
#WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'train_val_list/val_list/')

print("We are done")
