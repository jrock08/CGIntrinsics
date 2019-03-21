import time
import torch
import numpy as np
from options.test_options import TestOptions
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from models.models import create_model
from data.data_loader import CreateDataLoaderRelight
import torchvision
import pandas
from PIL import Image
from scipy import stats

from test_iiw import test_iiw
from test_relight import test_relight

opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
base_sub_name = opt.sub_name
results = []
for name in range(-14,15):
    if name < 0:
        opt.sub_name = base_sub_name + '_neg_%d'%(-name)
    else:
        opt.sub_name = base_sub_name + '_%d'%(name)

    model = create_model(opt)
    #print("WE ARE IN TESTING PHASE!!!!")
    #outp = test_relight(model, 'train_val_list/val_list/')
    out_relight = test_relight(model)
    out_iiw = test_iiw(model, 'test_list/', thresholds=opt.whdr_thresholds)

    df = pandas.DataFrame(out_relight, columns=['scene','img1','img2','score_relight','score_recon'])

    relight_mean = df[['score_relight','score_recon']].agg(stats.gmean)

    results.append([np.min(out_iiw[0]), relight_mean['score_relight'], relight_mean['score_recon']])

print 'whdr, relight, recon'
for res in results:
    print '{}, {}, {}'.format(*res)



#for epoch in range(17,25):
#    opt.which_epoch = epoch
#    model = create_model(opt)
#
#    #print("WE ARE IN TESTING PHASE!!!!")
#    #outp = test_relight(model, 'train_val_list/val_list/')
#    out_relight = test_relight(model)
#    out_iiw = test_iiw(model, 'test_list/', thresholds=opt.whdr_thresholds)
#
#    df = pandas.DataFrame(out_relight, columns=['scene','img1','img2','score_relight','score_recon'])
#
#    relight_mean = df[['score_relight','score_recon']].agg(stats.gmean)
#
#    results.append([np.min(out_iiw[0]), relight_mean['score_relight'], relight_mean['score_recon']])
#
#print 'whdr, relight, recon'
#for res in results:
#    print '{}, {}, {}'.format(*res)




