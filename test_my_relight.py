import time
import torch
import numpy as np
import sys, traceback
import h5py
from data.data_loader import CreateDataLoader
from data.data_loader import CreateDataLoaderRelight
import torchvision
from PIL import Image


def test_relight(model, full_root = '/data/jrock/Relighting_2019/', img_out_dir = ''):
    # print("============================= Validation ============================")
    model.switch_to_eval()

    outp = []
    data_loader_relight_test = CreateDataLoaderRelight(full_root + '/myImageCompositingInputs/', 1)
    #data_loader_relight_test = CreateDataLoaderRelight(full_root + '/BoyadzhievImageCompositingInputs/', 1)
    dataset_relight_test = data_loader_relight_test.load_data()

    written = set()
    selected_ids = ['img_0', 'img_6', 'img_7']
    for i, data in enumerate(dataset_relight_test):
        img1 = .9 * data[0] + .1 * data[1]
        img2 = .1 * data[0] + .9 * data[1]

        scene_id = data[2][0]
        img1_id = data[3][0]
        img2_id = data[4][0]
        if not (img1_id in selected_ids and img2_id in selected_ids):
            continue

        img1 = img1/img1.max()
        img2 = img2/img2.max()

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

        if img_out_dir is not '':
            if (scene_id, img1_id, img2_id) not in written:
                Image.fromarray(np.transpose(np.uint8(pred_R_rgb_1_print * 255.0), (1,2,0))).save(img_out_dir + '/reflectance_%s_%s_%s.png'%(scene_id, img1_id, img2_id))
                Image.fromarray(np.transpose(np.uint8(pred_S_1_print * 255.0), (1,2,0))).save(img_out_dir + '/shading_%s_%s_%s.png'%(scene_id, img1_id, img2_id))
                written.add((scene_id, img1_id))
            if (scene_id, img2_id, img1_id) not in written:
                Image.fromarray(np.transpose(np.uint8(pred_R_rgb_2_print * 255.0), (1,2,0))).save(img_out_dir + '/reflectance_%s_%s_%s.png'%(scene_id, img2_id, img1_id))
                Image.fromarray(np.transpose(np.uint8(pred_S_2_print * 255.0), (1,2,0))).save(img_out_dir + '/shading_%s_%s_%s.png'%(scene_id, img2_id, img1_id))
                written.add((scene_id, img2_id))
            Image.fromarray(np.transpose(np.uint8(img1[0] * 255.0), (1,2,0))).save(img_out_dir + '/img_%s_%s_%s.png'%(scene_id, img1_id, img2_id))
            Image.fromarray(np.transpose(np.uint8(img2[0] * 255.0), (1,2,0))).save(img_out_dir + '/img_%s_%s_%s.png'%(scene_id, img2_id, img1_id))



        img1_auto_recon = pred_R_rgb_1 * pred_S_1
        img2_auto_recon = pred_R_rgb_2 * pred_S_2

        img1_recon = pred_R_2 * pred_S_1
        img2_recon = pred_R_1 * pred_S_2

        img1_recon_rescale = img1_recon * img1.mean() / img1_recon.mean()
        img2_recon_rescale = img2_recon * img2.mean() / img2_recon.mean()

        img1_auto_recon_rescale = img1_auto_recon * img1.mean() / img1_auto_recon.mean()
        img2_auto_recon_rescale = img2_auto_recon * img2.mean() / img2_auto_recon.mean()

        #torch.mean((img1.cuda() - img1_recon)**2).cpu().item() + torch.mean((img2.cuda() - img2_recon)**2).cpu().item(),
        #torch.mean((img1.cuda() - img1_auto_recon)**2).cpu().item() + torch.mean((img2.cuda() - img2_auto_recon)**2).cpu().item(),
        #torch.mean((img1.cuda() - img1_recon_rescale)**2).cpu().item() + torch.mean((img2.cuda() - img2_recon_rescale)**2).cpu().item(),
        #torch.mean((pred_R_1 - pred_R_2)**2).item()])

        outp.append(scene_id.split('_') + [img1_id, img2_id] + [
        torch.mean(torch.abs(img1.cuda() - img1_recon)).cpu().item() + torch.mean(torch.abs(img2.cuda() - img2_recon)).cpu().item(),
        torch.mean(torch.abs(img1.cuda() - img1_auto_recon)).cpu().item() + torch.mean(torch.abs(img2.cuda() - img2_auto_recon)).cpu().item(),
        torch.mean(torch.abs(img1.cuda() - img1_recon_rescale)).cpu().item() + torch.mean(torch.abs(img2.cuda() - img2_recon_rescale)).cpu().item(),
        torch.mean(torch.abs(img1.cuda() - img1_auto_recon_rescale)).cpu().item() + torch.mean(torch.abs(img2.cuda() - img2_auto_recon_rescale)).cpu().item(),
        torch.mean(torch.abs(pred_R_1 - pred_R_2)).item() / (pred_R_2.max() - pred_R_2.min()).item()])
        #print outp

    return outp


if __name__ == '__main__':
    from options.test_options import TestOptions
    from models.models import create_model
    import pandas
    from scipy import stats
    import os

    opt = TestOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    model = create_model(opt)
    img_out_dir = '/data/jrock/out_relight_2019/' + opt.name + '_' + opt.sub_name + '/'

    if not os.path.exists(img_out_dir):
        os.mkdir(img_out_dir)


    #print("WE ARE IN TESTING PHASE!!!!")
    #outp = test_relight(model, 'train_val_list/val_list/')
    outp = test_relight(model, img_out_dir = img_out_dir)
    df = pandas.DataFrame(outp, columns=['scene','color','effect','img1','img2','score_relight','score_recon', 'score_relight_rescale', 'score_recon_rescale', 'score_consistency'])
    #df.to_pickle('scores_df.pkl')

    simple_results = df.mean()[['score_relight', 'score_recon', 'score_relight_rescale', 'score_recon_rescale', 'score_consistency']]

    # PER COLOR IN DEPTH RESULTS
    base = df[df['effect']=='empty']
    colors = df[df['effect']!='empty']
    comp = pandas.merge(base, colors, on='scene', suffixes=('_base',''))
    comp['relight_delta'] = comp['score_relight'] - comp['score_relight_base']

    matte_results = comp[comp['effect']=='matte'].groupby(['color']).mean()[['score_relight','score_recon','relight_delta','score_relight_rescale', 'score_recon_rescale', 'score_consistency']]
    glossy_results = comp[comp['effect']=='glossy'].groupby(['color']).mean()[['score_relight','score_recon', 'relight_delta','score_relight_rescale', 'score_recon_rescale', 'score_consistency']]


    with open('test_my_relight.txt','a') as f:
        f.write(opt.name + ' ' + opt.sub_name + '\n')
        f.write('simple:\n')
        f.write('{}\n'.format(simple_results))

        f.write('matte:\n')
        f.write('{}\n'.format(matte_results))

        f.write('glossy:\n')
        f.write('{}\n'.format(glossy_results))


    #for WHDR, WHDR_EQ, WHDR_INEQ in outp:
    #    print('WHDR %f'%WHDR)

    #WHDR, WHDR_EQ, WHDR_INEQ = test_relight(model, 'test_list/')
    #WHDR, WHDR_EQ, WHDR_INEQ = test_relight(model, 'train_val_list/val_list/')

    print("We are done")
