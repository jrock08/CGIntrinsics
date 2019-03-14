import time
import torch
import numpy as np
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from data.data_loader import CreateDataLoaderRelight
import torchvision
from PIL import Image


def test_relight(model, full_root = '/data/jrock/Relighting_2019/'):
    # print("============================= Validation ============================")
    model.switch_to_eval()

    outp = []
    #data_loader_relight_test = CreateDataLoaderRelight(full_root + '/myImageCompositingInputs/', 1)
    data_loader_relight_test = CreateDataLoaderRelight(full_root + '/BoyadzhievImageCompositingInputs/', 1)
    dataset_relight_test = data_loader_relight_test.load_data()

    written = set()
    for i, data in enumerate(dataset_relight_test):
        img1 = .9 * data[0] + .1 * data[1]
        img2 = .1 * data[0] + .9 * data[1]

        scene_id = data[2][0]
        img1_id = data[3][0]
        img2_id = data[4][0]
        pred_R_1, pred_R_rgb_1, pred_S_1 = model.get_output_images(img1)
        pred_R_2, pred_R_rgb_2, pred_S_2 = model.get_output_images(img2)

        pred_S_1_print = pred_S_1[0].cpu().detach().numpy()
        p1_m = np.percentile(pred_S_1_print, 90)
        pred_S_1_print = np.clip(pred_S_1_print / p1_m, 0, 1)
        pred_S_1_print = np.repeat(pred_S_1_print, 3, 0)
        pred_R_rgb_1_print = pred_R_rgb_1[0].cpu().detach().numpy()
        pred_R_rgb_1_print = np.clip(pred_R_rgb_1_print / np.percentile(pred_R_rgb_1_print, 99),0,1)
        pred_S_2_print = pred_S_2[0].cpu().detach().numpy()
        p2_m = np.percentile(pred_S_2_print, 90)
        pred_R_rgb_2_print = pred_R_rgb_2[0].cpu().detach().numpy()
        pred_R_rgb_2_print = np.clip(pred_R_rgb_2_print / np.percentile(pred_R_rgb_2_print, 99),0,1)
        pred_S_2_print = np.clip(pred_S_2_print / p2_m, 0, 1)
        pred_S_2_print = np.repeat(pred_S_2_print, 3, 0)

        if (scene_id, img1_id, img2_id) not in written:
            Image.fromarray(np.transpose(np.uint8(pred_R_rgb_1_print * 255.0), (1,2,0))).save('/data/jrock/out_relight_2019/reflectance_%s_%s_%s.png'%(scene_id, img1_id, img2_id))
            Image.fromarray(np.transpose(np.uint8(pred_S_1_print * 255.0), (1,2,0))).save('/data/jrock/out_relight_2019/shading_%s_%s_%s.png'%(scene_id, img1_id, img2_id))
            written.add((scene_id, img1_id))
        if (scene_id, img2_id, img1_id) not in written:
            Image.fromarray(np.transpose(np.uint8(pred_R_rgb_2_print * 255.0), (1,2,0))).save('/data/jrock/out_relight_2019/reflectance_%s_%s_%s.png'%(scene_id, img2_id, img1_id))
            Image.fromarray(np.transpose(np.uint8(pred_S_2_print * 255.0), (1,2,0))).save('/data/jrock/out_relight_2019/shading_%s_%s_%s.png'%(scene_id, img2_id, img1_id))
            written.add((scene_id, img2_id))

        img1_auto_recon = pred_R_rgb_1 * pred_S_1
        img2_auto_recon = pred_R_rgb_2 * pred_S_2

        img1_recon = pred_R_2 * pred_S_1
        img2_recon = pred_R_1 * pred_S_2

        outp.append([scene_id, img1_id, img2_id] + [torch.mean((img1.cuda() - img1_recon)**2).cpu().item() + torch.mean((img2.cuda() - img2_recon)**2).cpu().item(),
        torch.mean((img1.cuda() - img1_auto_recon)**2).cpu().item() + torch.mean((img2.cuda() - img2_auto_recon)**2).cpu().item()])
        #print outp

    return outp


if __name__ == '__main__':
    from options.test_options import TestOptions
    from models.models import create_model
    import pandas
    from scipy import stats

    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)

    #print("WE ARE IN TESTING PHASE!!!!")
    #outp = test_relight(model, 'train_val_list/val_list/')
    outp = test_relight(model)
    df = pandas.DataFrame(outp, columns=['scene','img1','img2','score_relight','score_recon'])
    df.to_pickle('scores_df.pkl')

    for img_id in range(10):
        Z = df[(df['img1'] == 'img_%d'%(img_id)) | (df['img2'] == 'img_%d'%(img_id))].groupby('scene')['score_relight','score_recon'].agg(stats.gmean)
        print 'img_contribution: %d'%(img_id)
        print Z

    print 'overall:'
    print df.groupby('scene')[['score_relight','score_recon']].agg(stats.gmean)

    #for WHDR, WHDR_EQ, WHDR_INEQ in outp:
    #    print('WHDR %f'%WHDR)

    #WHDR, WHDR_EQ, WHDR_INEQ = test_relight(model, 'test_list/')
    #WHDR, WHDR_EQ, WHDR_INEQ = test_relight(model, 'train_val_list/val_list/')

    print("We are done")
