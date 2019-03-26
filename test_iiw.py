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
from PIL import Image



def test_iiw(model, list_name, full_root = '/data/jrock/IIW_2019/', thresholds = None, img_out_dir = ''):
    # print("============================= Validation ============================")
    model.switch_to_eval()

    outp = []
    #thresholds = np.linspace(0,1,15)
    if thresholds is None:
        thresholds = [.5]
    #thresholds = [.22]
    total_loss = np.zeros(len(thresholds))
    total_loss_eq = np.zeros(len(thresholds))
    total_loss_ineq = np.zeros(len(thresholds))
    total_count = 0.0
    img_count = 0
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
            [total_whdr, total_whdr_eq, total_whdr_ineq, count], pred_rgb, pred_S = model.evlaute_iiw(stacked_img, targets, thresholds)

            if img_out_dir != '':
                for i in range(pred_rgb.shape[0]):
                    pred_R_print = pred_rgb[i].detach().cpu().numpy()
                    pred_S_print = pred_S[i].detach().cpu().numpy()
                    p1_m = np.percentile(pred_S_print, 90)
                    pred_S_print = np.clip(pred_S_print / p1_m, 0, 1)
                    pred_S_print = np.repeat(pred_S_print, 3, 0)
                    pred_R_print = np.clip(pred_R_print / np.percentile(pred_R_print, 99),0,1)

                    Image.fromarray(np.transpose(np.uint8(pred_R_print * 255.0), (1,2,0))).save(img_out_dir + '/reflectance_%d.png'%(img_count))
                    Image.fromarray(np.transpose(np.uint8(pred_S_print * 255.0), (1,2,0))).save(img_out_dir + '/shading_%d.png'%(img_count))
                    Image.fromarray(np.transpose(np.uint8(stacked_img[i] * 255.0), (1,2,0))).save(img_out_dir + '/img_%d.png'%(img_count))
                    img_count += 1


            total_loss += total_whdr
            total_loss_eq += total_whdr_eq
            total_loss_ineq += total_whdr_ineq

            total_count += count
            print("Testing WHDR error ", j, i, total_loss/total_count)

    return total_loss/(total_count), total_loss_eq/total_count, total_loss_ineq/total_count

if __name__ == '__main__':
    import os
    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    model = create_model(opt)
    img_out_dir = '/data/jrock/out_relight_2019/iiw/' + opt.name + '_' + opt.sub_name + '/'
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    #print("WE ARE IN TESTING PHASE!!!!")
    #outp = test_iiw(model, 'train_val_list/val_list/')
    outp = test_iiw(model, 'test_list/', thresholds=opt.whdr_thresholds, img_out_dir=img_out_dir)
    with open('val_whdr.txt','a') as f:
        f.write(opt.name + ' ' + opt.sub_name)
        f.write('WHDR {}\n'.format(outp[0]))
        f.write('WHDR_EQ {}'.format(outp[1]))
        f.write('WHDR_INEQ {}'.format(outp[2]))
    print('WHDR {}'.format(outp[0]))
    print('WHDR_EQ {}'.format(outp[1]))
    print('WHDR_INEQ {}'.format(outp[2]))
    #for WHDR, WHDR_EQ, WHDR_INEQ in outp:
    #    print('WHDR %f'%WHDR)

    #WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'test_list/')
    #WHDR, WHDR_EQ, WHDR_INEQ = test_iiw(model, 'train_val_list/val_list/')

    print("We are done")
