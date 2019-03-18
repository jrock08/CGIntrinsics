import h5py
import torch.utils.data as data
import pickle
import PIL
import numpy as np
import torch
import string
from scipy import misc
import os
import os.path
import sys
import math, random
import json

import skimage
from skimage import io
from skimage.transform import resize
from skimage.transform import rescale
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle
from PIL import Image
from scipy.ndimage.measurements import label
import scipy.io as sio

def make_dataset(list_dir):
    file_name = list_dir + "img_batch.p"
    images_list = pickle.load( open( file_name, "rb" ) )

    return images_list

def rgb_to_irg(rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 2] = s / 3.0
    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    return irg

def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_chromaticity(rgb):
    """ converts rgb to chromaticity """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1) + 1e-6

    irg[..., 0] = rgb[..., 0] / s
    irg[..., 1] = rgb[..., 1] / s
    irg[..., 2] = rgb[..., 2] / s

    return irg

class CGIntrinsicsImageFolder(data.Dataset):

    def __init__(self, root, list_dir, transform=None,
                 loader=None, clamp_pairs=-1):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.img_list = img_list
        self.transform = transform
        self.loader = loader
        self.num_scale  = 4
        self.sigma_n = 0.02
        self.half_window = 1
        self.height = 384
        self.width = 512
        self.original_h = 480
        self.original_w = 640
        self.rotation_range = 5.0
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        self.stat_dict = {}
        f = open(self.root + "/CGIntrinsics/intrinsics_final/rgbe_image_stats.txt","r")
        line = f.readline()
        while line:
            line = line.split()
            self.stat_dict[line[0]] = float(line[2])
            line = f.readline()
        f.close()
        self.clamp_pairs = clamp_pairs

    def DA(self, img, mode, random_pos, random_filp):

        if random_filp > 0.5:
            img = np.fliplr(img)

        # img = rotate(img,random_angle, order = mode)
        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        img = resize(img, (self.height, self.width), order = mode)

        return img

    def construst_S_weights(self, normal):

        center_feature = np.repeat( np.expand_dims(normal[4,:,:,:], axis =0), 9, axis = 0)
        feature_diff = center_feature - normal

        r_w = np.exp( - np.sum( feature_diff[:,:,:,0:3]**2  , 3) / (self.sigma_n**2))

        return r_w

    def construst_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,h-2,w-2, 3))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :]
                ct_idx += 1

        return sub_C

    def load_CGIntrinsics(self, path):
        import time
        st_total = time.time()

        st = time.time()
        img_path = self.root + "/CGIntrinsics/intrinsics_final/images/" + path
        srgb_img = np.float32(io.imread(img_path))/ 255.0
        file_name = path.split('/')

        R_path = self.root + "/CGIntrinsics/intrinsics_final/images/" + file_name[0] + "/" + file_name[1][:-4] + "_albedo.png"
        gt_R = np.float32(io.imread(R_path))/ 255.0

        mask_path = self.root + "/CGIntrinsics/intrinsics_final/images/" + file_name[0] + "/" + file_name[1][:-4] + "_mask.png"
        mask = np.float32(io.imread(mask_path))/ 255.0
        en = time.time()
        im_load_time = en-st

        gt_R_gray = np.mean(gt_R, 2)
        mask[gt_R_gray < 1e-6] = 0
        mask[np.mean(srgb_img,2) < 1e-6] = 0

        st = time.time()
        mask = skimage.morphology.binary_erosion(mask, square(11))
        mask = np.expand_dims(mask, axis = 2)
        mask = np.repeat(mask, 3, axis= 2)
        gt_R[gt_R <1e-6] = 1e-6
        en = time.time()
        morph_time = en-st


        # do normal DA
        # random_angle = random.random() * self.rotation_range * 2.0 - self.rotation_range # random angle between -5 --- 5 degree
        random_filp = random.random()
        random_start_y = random.randint(0, 9)
        random_start_x = random.randint(0, 9)

        random_pos = [random_start_y, random_start_y + self.original_h - 10, random_start_x, random_start_x + self.original_w - 10]
        st = time.time()
        srgb_img = self.DA(srgb_img, 1, random_pos, random_filp)
        gt_R = self.DA(gt_R, 1,  random_pos, random_filp)
        # cam_normal = self.DA(cam_normal, 0,  random_pos, random_filp)
        mask = self.DA(mask, 0,  random_pos, random_filp)
        rgb_img = srgb_img**2.2
        gt_S = rgb_img / gt_R
        en = time.time()
        DA_time = en - st

        st = time.time()
        search_name = path[:-4] + ".rgbe"
        irridiance = self.stat_dict[search_name]

        if irridiance < 0.25:
            srgb_img = denoise_tv_chambolle(srgb_img, weight=0.05, multichannel=True)
            gt_S = denoise_tv_chambolle(gt_S, weight=0.1, multichannel=True)
        en = time.time()
        irridiance_time = en-st

        mask[gt_S > 10] = 0
        gt_S[gt_S > 20] = 20
        mask[gt_S < 1e-4] = 0
        gt_S[gt_S < 1e-4] = 1e-4

        if np.sum(mask) < 10:
            max_S = 1.0
        else:
            max_S = np.percentile(gt_S[mask > 0.5], 90)

        gt_S = gt_S/max_S

        gt_S = np.mean(gt_S, 2)
        gt_S = np.expand_dims(gt_S, axis = 2)

        gt_R_gray = np.mean(gt_R, 2)
        gt_R_gray = np.expand_dims(gt_R_gray, axis = 2)
        en_total = time.time()
        #print '{}: im_load: {} morph: {} DA: {} irridiance: {} total: {}'.format(path, im_load_time, morph_time, DA_time, irridiance_time, en_total - st_total)

        return srgb_img, gt_R_gray, gt_R, gt_S, mask, random_filp

    def CGIntrinsics_pair(self, path, gt_albedo, random_filp):
        import time

        st = time.time()
        super_pixel_path = self.root + "/CGIntrinsics/intrinsics_final/superpixels/" + path + ".mat"
        super_pixel_mat = sio.loadmat(super_pixel_path)
        super_pixel_mat = super_pixel_mat['data']
        en = time.time()
        load_time = en-st

        st = time.time()
        final_list = []
        for i in range(len(super_pixel_mat)):
            pos = super_pixel_mat[i][0]

            if pos.shape[0] < 2:
                continue

            rad_idx = random.randint(0, pos.shape[0]-1)
            final_list.append( (pos[rad_idx,0], pos[rad_idx,1]) )
        en = time.time()
        final_list_time = en-st

        if self.clamp_pairs > 0 and len(final_list) > self.clamp_pairs:
            final_list = random.sample(final_list, self.clamp_pairs)

        final_list_len = len(final_list)

        st = time.time()
        eq_list = []
        ineq_list = []

        row = gt_albedo.shape[0]
        col = gt_albedo.shape[1]

        for i in range(0, len(final_list)-1):
            y_1, x_1 = final_list[i]

            y_1 = int(y_1*row)
            x_1 = int(x_1*col)

            if random_filp:
                x_1 = col - 1 - x_1

            v_1 = gt_albedo[y_1, x_1]

            if v_1 < 2e-4:
                continue

            for j in range(i+1, len(final_list)):
                y_2, x_2 = final_list[j]

                y_2 = int(y_2*row)
                x_2 = int(x_2*col)

                if random_filp:
                    x_2 = col - 1 - x_2

                v_2 = gt_albedo[y_2, x_2]

                if v_2 < 2e-4:
                    continue

                ratio = v_1 / v_2

                if ratio < 1.05 and ratio > 1./1.05:
                    eq_list.append([y_1, x_1, y_2, x_2])
                elif ratio > 1.5:
                    ineq_list.append([y_1, x_1, y_2, x_2])
                elif ratio < 1./1.5:
                    ineq_list.append([y_2, x_2, y_1, x_1])
        en = time.time()
        ineq_list_time = en-st

        eq_mat = np.asarray(eq_list)
        ineq_mat = np.asarray(ineq_list)

        #print 'load_time: {}, final_list_time: {}, ineq_list_time: {}, list_len: {} eq_ineq_list_len: {}'.format(load_time, final_list_time, ineq_list_time, final_list_len, len(eq_list) + len(ineq_list))

        if eq_mat.shape[0] > 0:
            eq_mat = torch.from_numpy(eq_mat).contiguous().float()
        else:
            eq_mat = torch.Tensor(1,1)


        if ineq_mat.shape[0] > 0:
            ineq_mat = torch.from_numpy(ineq_mat).contiguous().float()
        else:
            ineq_mat = torch.Tensor(1,1)


        return eq_mat, ineq_mat


    def __getitem__(self, index):
        targets_1 = {}
        img_path = self.img_list[index]
        # split_img_path = img_path.split('/')
        full_path = self.root + img_path

        srgb_img, gt_R, gt_R_color, gt_S, mask, random_filp = self.load_CGIntrinsics(img_path)

        targets_1['CGIntrinsics_ordinal_path'] = img_path
        targets_1['random_filp'] = random_filp > 0.5

        rgb_img = srgb_img**2.2
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()

        targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()
        final_img = torch.from_numpy(np.transpose(srgb_img, (2, 0, 1))).contiguous().float()
        targets_1['mask'] = torch.from_numpy(np.transpose(mask, (2 , 0 ,1))).contiguous().float()
        targets_1['gt_R'] = torch.from_numpy(np.transpose(gt_R, (2 , 0 ,1))).contiguous().float()
        targets_1['gt_R_color'] = torch.from_numpy(np.transpose(gt_R_color, (2,0,1))).contiguous().float()
        targets_1['gt_S'] = torch.from_numpy(np.transpose(gt_S, (2 , 0 ,1))).contiguous().float()
        targets_1['path'] = full_path

        sparse_path_1s = self.root + "/CGIntrinsics/intrinsics_final/sparse_hdf5_S/384x512/R0.h5"

        eq_mat, ineq_mat = self.CGIntrinsics_pair(targets_1['CGIntrinsics_ordinal_path'], targets_1['gt_R'][0,:,:], targets_1['random_filp'])
        targets_1['eq_mat'] = eq_mat
        targets_1['ineq_mat'] = ineq_mat

        return final_img, targets_1, sparse_path_1s

    def __len__(self):
        return len(self.img_list)


class Render_ImageFolder(data.Dataset):
    def __init__(self, root, list_dir, transform=None,
                 loader=None):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.img_list = img_list
        self.transform = transform
        self.loader = loader
        self.num_scale  = 4
        self.sigma_I = 0.1
        self.half_window = 1
        self.height = 384
        self.width = 512
        self.original_h = 480
        self.original_w = 640
        self.rotation_range = 5.0
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        self.sigma_chro = 0.025

    def DA(self, img, mode, random_pos, random_filp, h, w):

        if random_filp > 0.5:
            img = np.fliplr(img)

        # img = rotate(img,random_angle, order = mode)
        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        img = resize(img, (h, w), order = mode)

        return img

    def construst_R_weights(self, N_feature):

        center_feature = np.repeat( np.expand_dims(N_feature[4, :, :,:], axis =0), 9, axis = 0)
        feature_diff = center_feature - N_feature

        r_w = np.exp( - np.sum( feature_diff[:,:,:,0:3]**2  , 3) / (self.sigma_chro**2))

        return r_w

    def construst_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,h-2,w-2, 3))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :]
                ct_idx += 1

        return sub_C


    def load_Render(self, path):
        img_name = path
        img_path = self.root + "/CGIntrinsics/intrinsics_final/rendered/images/" + path
        rgb_img = np.float32(io.imread(img_path))/ 255.0

        if rgb_img.shape[2] == 4:
            print("=================rgb_img_path ", img_path)
            rgb_img = rgb_img[:,:,0:3]

        srgb_img = rgb_img**(1.0/2.2)


        R_path = self.root + "/CGIntrinsics/intrinsics_final/rendered/albedo/" + path[:-4].split('_')[0] + "_albedo.png"
        gt_R = np.float32(io.imread(R_path))/ 255.0

        if gt_R.shape[2] == 4:
            print("=================gt_R_path ", img_path)
            gt_R = gt_R[:,:,0:3]

        gt_R[gt_R <1e-4] = 1e-4

        mask_path = self.root + "/CGIntrinsics/intrinsics_final/rendered/mask/" + path[:-4].split('_')[0] + "_alpha.png"
        mask = np.float32(io.imread(mask_path))/ 255.0

        if mask.shape[2] == 4:
            print("=================mask_path ", img_path)
            mask = mask[:,:,0:3]

        mask = np.mean(mask, 2)
        mask[mask < 0.99] = 0
        mask = skimage.morphology.erosion(mask, square(7))
        mask = np.expand_dims(mask, axis = 2)
        mask = np.repeat(mask, 3, axis= 2)

        # do normal DA
        # random_angle = random.random() * self.rotation_range * 2.0 - self.rotation_range # random angle between -5 --- 5 degree
        random_filp = random.random()
        random_start_y = random.randint(0, 19)
        random_start_x = random.randint(0, 19)
        random_pos = [random_start_y, random_start_y + srgb_img.shape[0] - 20, random_start_x, random_start_x + srgb_img.shape[1] - 20]

        ratio = float(srgb_img.shape[0])/float(srgb_img.shape[1])

        if ratio > 1.73:
            h, w = 512, 256
        elif ratio < 1.0/1.73:
            h, w = 256, 512
        elif ratio > 1.41:
            h, w = 768, 512
        elif ratio < 1./1.41:
            h, w = 512, 768
        elif ratio > 1.15:
            h, w = 512, 384
        elif ratio < 1./1.15:
            h, w = 384, 512
        else:
            h, w = 512, 512

        srgb_img = self.DA(srgb_img, 1, random_pos, random_filp, h, w)
        gt_R = self.DA(gt_R, 1,  random_pos, random_filp, h, w)
        # cam_normal = self.DA(cam_normal, 0,  random_pos, random_filp)
        mask = self.DA(mask, 0,  random_pos, random_filp, h, w)

        rgb_img = srgb_img**2.2

        gt_S = rgb_img / gt_R

        mask[gt_S > 20] = 0
        gt_S[gt_S > 20] = 20
        gt_S[gt_S < 1e-4] = 1e-4
        gt_S[mask == 0] = 0.5

        if np.sum(mask) < 10:
            max_S = 1.0
        else:
            max_S = np.percentile(gt_S[mask > 0.5], 90)

        gt_S = gt_S/max_S

        gt_S = np.mean(gt_S, 2)
        gt_S = np.expand_dims(gt_S, axis = 2)

        gt_R = np.mean(gt_R,2)
        gt_R = np.expand_dims(gt_R, axis = 2)

        return srgb_img, gt_R, gt_S, mask

    def __getitem__(self, index):
        targets_1 = {}
        img_path = self.img_list[index]
        # split_img_path = img_path.split('/')
        full_path = self.root + img_path
        srgb_img, gt_R, gt_S, mask = self.load_Render(img_path)

        rgb_img = srgb_img**2.2
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()
        targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()
        final_img = torch.from_numpy(np.transpose(srgb_img, (2, 0, 1))).contiguous().float()
        targets_1['mask'] = torch.from_numpy(np.transpose(mask, (2 , 0 ,1))).contiguous().float()
        targets_1['gt_R'] = torch.from_numpy(np.transpose(gt_R, (2 , 0 ,1))).contiguous().float()
        targets_1['gt_S'] = torch.from_numpy(np.transpose(gt_S, (2 , 0 ,1))).contiguous().float()
        targets_1['path'] = full_path

        return {'img_1': final_img, 'target_1': targets_1}

    def __len__(self):
        return len(self.img_list)



class IIW_ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, mode, is_flip, transform=None,
                 loader=None, load_long_range_annotes = True, clamp_pairs=-1):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.is_flip = is_flip
        self.img_list = img_list
        # self.targets_list = targets_list
        # self.img_list_2 = img_list_2
        self.transform = transform
        self.loader = loader
        self.num_scale  = 4
        self.sigma_chro = 0.025
        self.sigma_I = 0.1
        self.half_window = 1
        self.current_o_idx = mode
        self.set_o_idx(mode)
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        self.load_long_range_annotes = load_long_range_annotes
        self.clamp_pairs = clamp_pairs

    def set_o_idx(self, o_idx):
        self.current_o_idx = o_idx

        if o_idx == 0:
            self.height = 256
            self.width = 384
        elif o_idx == 1:
            self.height = 384
            self.width = 256
        elif o_idx == 2:
            self.height = 384
            self.width = 384
        elif o_idx == 3:
            self.height = 384
            self.width = 512
        else:
            self.height = 512
            self.width = 384

    def DA(self, img, mode, random_filp):

        # if random_filp > 0.5:
            # img = np.fliplr(img)

        # img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]

        img = resize(img, (self.height, self.width), order = mode)

        return img

    def iiw_loader(self, img_path):
        img_path = img_path[-1][:-3]
        img_path = self.root + "CGIntrinsics/IIW/data/" + img_path
        img = np.float32(io.imread(img_path))/ 255.0
        oringinal_shape = img.shape

        img = resize(img, (self.height, self.width))

        random_filp = random.random()

        if self.is_flip and random_filp > 0.5:
            img = np.fliplr(img)

        return img, random_filp, oringinal_shape

    def construst_R_weights(self, N_feature):

        center_feature = np.repeat( np.expand_dims(N_feature[4, :, :,:], axis =0), 9, axis = 0)
        feature_diff = center_feature - N_feature

        r_w = np.exp( - np.sum( feature_diff[:,:,:,0:2]**2  , 3) / (self.sigma_chro**2)) \
                    * np.exp(- (feature_diff[:,:,:,2]**2) /(self.sigma_I**2) )

        return r_w

    def construst_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,h-2,w-2, 3))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :]
                ct_idx += 1

        return sub_C

    def parse_true_judgements(self, judgements):
        #x,y in normalized size
        #eq_matrix = [y1,x1,y2,x2,weight]

        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}

        eq_list = []
        ineq_list = []

        for c in comparisons:
            darker = c['darker']

            if darker not in ('1','2','E'):
                continue

            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            p1 = id_to_points[c['point1']]
            p2 = id_to_points[c['point2']]

            if not p1['opaque'] or not p2['opaque']:
                continue

            if darker == 'E':
                eq_list.append([p1['y'],p1['x'],p2['y'],p2['x'], weight])
            elif darker == '2':
                ineq_list.append([p1['y'],p1['x'],p2['y'],p2['x'], weight])
            else:
                ineq_list.append([p2['y'],p2['x'],p1['y'],p1['x'], weight])

        if len(eq_list) > 0:
            equal_mat = torch.from_numpy(np.array(eq_list)).contiguous().float()
        else:
            equal_mat = torch.Tensor(1,1)

        if len(ineq_list) > 0:
            inequal_mat = torch.from_numpy(np.array(ineq_list)).contiguous().float()
        else:
            inequal_mat = torch.Tensor(1,1)

        return equal_mat, inequal_mat

    def long_range_loader(self, h5_path):
        hdf5_file_read_img = h5py.File(h5_path,'r')
        num_eq = hdf5_file_read_img.get('/info/num_eq')
        num_eq = np.float32(np.array(num_eq))
        num_eq = int(num_eq[0][0])

        if num_eq > 0:
            equal_mat = hdf5_file_read_img.get('/info/equal')
            equal_mat = np.float32(np.array(equal_mat))
            equal_mat = np.transpose(equal_mat, (1, 0))
            equal_mat = torch.from_numpy(equal_mat).contiguous().float()
        else:
            equal_mat = torch.Tensor(1,1)

        num_ineq = hdf5_file_read_img.get('/info/num_ineq')
        num_ineq = np.float32(np.array(num_ineq))
        num_ineq = int(num_ineq[0][0])

        if num_ineq > 0:
            ineq_mat = hdf5_file_read_img.get('/info/inequal')
            ineq_mat = np.float32(np.array(ineq_mat))
            ineq_mat = np.transpose(ineq_mat, (1, 0))
            ineq_mat = torch.from_numpy(ineq_mat).contiguous().float()
        else:
            ineq_mat = torch.Tensor(1,1)

        hdf5_file_read_img.close()

        if self.clamp_pairs > 0 and equal_mat.shape[0] > self.clamp_pairs:
            sample = random.sample(range(equal_mat.shape[0]), self.clamp_pairs)
            equal_mat = equal_mat.index_select(0, torch.tensor(sample, requires_grad=False))

        if self.clamp_pairs > 0 and ineq_mat.shape[0] > self.clamp_pairs:
            sample = random.sample(range(ineq_mat.shape[0]), self.clamp_pairs)
            ineq_mat.index_select(0, torch.tensor(sample, requires_grad=False))

        return equal_mat, ineq_mat


    def __getitem__(self, index):

        targets_1 = {}
        # temp_targets = {}

        img_path = self.root + "/CGIntrinsics/IIW/" + self.img_list[self.current_o_idx][index]
        judgement_path = self.root + "/CGIntrinsics/IIW/data/" + img_path.split('/')[-1][0:-6] + 'json'

        mat_path = self.root + "/CGIntrinsics/IIW/long_range_data_4/" + img_path.split('/')[-1][0:-6] + "h5"
        targets_1['mat_path'] = mat_path

        # img, random_filp = self.iiw_loader(img_path)
        srgb_img, random_filp, oringinal_shape = self.iiw_loader(self.img_list[self.current_o_idx][index].split('/'))

        targets_1['path'] = "/" + img_path.split('/')[-1]
        targets_1["judgements_path"] = judgement_path
        targets_1["random_filp"] = random_filp > 0.5
        targets_1["oringinal_shape"] = oringinal_shape

        # if random_filp > 0.5:
            # sparse_path_1r = self.root + "/IIW/iiw-dataset/sparse_hdf5_batch_flip/" + img_path.split('/')[-1] + "/R0.h5"
        # else:
            # sparse_path_1r = self.root + "/IIW/iiw-dataset/sparse_hdf5_batch/" + img_path.split('/')[-1] + "/R0.h5"

        rgb_img = srgb_to_rgb(srgb_img)
        rgb_img[rgb_img < 1e-4] = 1e-4
        chromaticity = rgb_to_chromaticity(rgb_img)
        targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()
        targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()

        for i in range(0, self.num_scale):
            feature_3d = rgb_to_irg(rgb_img)
            sub_matrix = self.construst_sub_matrix(feature_3d)
            r_w = self.construst_R_weights(sub_matrix)
            targets_1['r_w_s'+ str(i)] = torch.from_numpy(r_w).float()
            rgb_img = rgb_img[::2,::2,:]


        final_img = torch.from_numpy(np.ascontiguousarray(np.transpose(srgb_img, (2,0,1)))).contiguous().float()

        if self.load_long_range_annotes:
            eq_mat, ineq_mat = self.long_range_loader(targets_1['mat_path'])
            targets_1['eq_mat'] = eq_mat
            targets_1['ineq_mat'] = ineq_mat

        judgements = json.load(open(targets_1['judgements_path']))
        true_eq_mat, true_ineq_mat = self.parse_true_judgements(judgements)
        targets_1['gt_eq_mat'] = true_eq_mat
        targets_1['gt_ineq_mat'] = true_ineq_mat

        sparse_shading_name = str(self.height) + "x" + str(self.width)

        if self.current_o_idx == 0:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 1:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name +  "/R0.h5"
        elif self.current_o_idx == 2:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 3:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 4:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"

        return final_img, targets_1, sparse_path_1s


    def __len__(self):
        return len(self.img_list[self.current_o_idx])



class SAW_ImageFolder(data.Dataset):

    def __init__(self, root, list_dir, mode, is_flip, transform=None,
                 loader=None):
        # load image list from hdf5
        img_list = make_dataset(list_dir)
        if len(img_list) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.list_dir = list_dir
        self.is_flip = is_flip
        self.img_list = img_list
        # self.targets_list = targets_list
        # self.img_list_2 = img_list_2
        self.transform = transform
        self.loader = loader
        self.num_scale  = 4
        self.sigma_chro = 0.025
        self.sigma_I = 0.1
        self.half_window = 1
        self.current_o_idx = mode
        self.set_o_idx(mode)
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        self.pixel_labels_dir = self.root + "/CGIntrinsics/SAW/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0"

    def set_o_idx(self, o_idx):
        self.current_o_idx = o_idx

        if o_idx == 0:
            self.height = 256
            self.width = 384
        elif o_idx == 1:
            self.height = 384
            self.width = 256
        elif o_idx == 2:
            self.height = 384
            self.width = 384
        elif o_idx == 3:
            self.height = 384
            self.width = 512
        else:
            self.height = 512
            self.width = 384

    def DA(self, img, mode, random_pos, random_filp):

        if random_filp > 0.5:
            img = np.fliplr(img)

        # img = rotate(img,random_angle, order = mode)
        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        img = resize(img, (self.height, self.width), order = mode)

        return img

    def load_img_arr(self, photo_id):
        img_path = self.root + "/CGIntrinsics/SAW/saw_images_512/" + str(photo_id) + ".png"
        srg_img = Image.open(img_path)
        srg_img = np.asarray(srg_img).astype(float) / 255.0
        return srg_img

    def load_pixel_labels(self, pixel_labels_dir, photo_id):
        pixel_labels_path = os.path.join(pixel_labels_dir, '%s.npy' % photo_id)
        if not os.path.exists(pixel_labels_path):
            raise ValueError('Could not find ground truth labels at "%s"' % pixel_labels_path)

        return np.load(pixel_labels_path)

    def saw_loader(self, photo_id):
        # img_path = self.root + "/SAW/saw_release/saw/saw_images_512/" + str(photo_id) + ".png"
        img = self.load_img_arr(photo_id)
        saw_mask = self.load_pixel_labels(self.pixel_labels_dir, photo_id)

        return img, saw_mask

    def construst_R_weights(self, N_feature):

        center_feature = np.repeat( np.expand_dims(N_feature[4, :, :,:], axis =0), 9, axis = 0)
        feature_diff = center_feature - N_feature

        r_w = np.exp( - np.sum( feature_diff[:,:,:,0:2]**2  , 3) / (self.sigma_chro**2)) \
                    * np.exp(- (feature_diff[:,:,:,2]**2) /(self.sigma_I**2) )

        return r_w

    def construst_sub_matrix(self, C):
        h = C.shape[0]
        w = C.shape[1]

        sub_C = np.zeros( (9 ,h-2,w-2, 3))
        ct_idx = 0
        for k in range(0, self.half_window*2+1):
            for l in range(0,self.half_window*2+1):
                sub_C[ct_idx,:,:,:] = C[self.half_window + self.Y[k,l]:h- self.half_window + self.Y[k,l], \
                self.half_window + self.X[k,l]: w-self.half_window + self.X[k,l] , :]
                ct_idx += 1

        return sub_C


    def __getitem__(self, index):
        targets_1 = {}
        # temp_targets = {}

        photo_id = self.img_list[self.current_o_idx][index]

        srgb_img, saw_mask = self.saw_loader(photo_id)
        targets_1['path'] = str(photo_id)

        saw_mask_0 = (saw_mask == 0)
        saw_mask_0 = skimage.morphology.binary_dilation(saw_mask_0, square(9)).astype(np.float32)
        saw_mask_0 = 1 - saw_mask_0

        saw_mask_1 = (saw_mask == 1)
        saw_mask_1 = skimage.morphology.binary_dilation(saw_mask_1, square(9))

        saw_mask_2 = (saw_mask == 2)

        saw_mask_0 = np.expand_dims(saw_mask_0, axis =2)
        saw_mask_1 = np.expand_dims(saw_mask_1, axis =2)
        saw_mask_2 = np.expand_dims(saw_mask_2, axis =2)

        original_h = srgb_img.shape[0]
        original_w = srgb_img.shape[1]

        random_filp = random.random()
        random_start_y = random.randint(0, 9)
        random_start_x = random.randint(0, 9)
        random_pos = [random_start_y, random_start_y + original_h - 10, random_start_x, random_start_x + original_w - 10]

        srgb_img = self.DA(srgb_img, 1, random_pos, random_filp)
        saw_mask_0 = self.DA(saw_mask_0, 0,  random_pos, random_filp)
        saw_mask_1 = self.DA(saw_mask_1, 0,  random_pos, random_filp)
        saw_mask_2 = self.DA(saw_mask_2, 0,  random_pos, random_filp)

        saw_mask_1, num_mask_1 = label(saw_mask_1)

        saw_mask_1 = saw_mask_1.astype(np.float32)

        saw_mask_2, num_mask_2 = label(saw_mask_2)
        saw_mask_2 = saw_mask_2.astype(np.float32)

        targets_1["num_mask_1"] = num_mask_1
        targets_1["num_mask_2"] = num_mask_2

        targets_1["saw_mask_0"] = torch.from_numpy( np.transpose(saw_mask_0, (2,0,1)) ).contiguous().float()
        targets_1["saw_mask_1"] = torch.from_numpy( np.transpose(saw_mask_1, (2,0,1)) ).contiguous().float()
        targets_1["saw_mask_2"] = torch.from_numpy( np.transpose(saw_mask_2, (2,0,1)) ).contiguous().float()

        rgb_img = srgb_to_rgb(srgb_img)
        rgb_img[rgb_img < 1e-4] = 1e-4

        chromaticity = rgb_to_chromaticity(rgb_img)
        targets_1['chromaticity'] = torch.from_numpy(np.transpose(chromaticity, (2,0,1))).contiguous().float()
        targets_1["rgb_img"] = torch.from_numpy(np.transpose(rgb_img, (2,0,1))).contiguous().float()

        for i in range(0, self.num_scale):
            feature_3d = rgb_to_irg(rgb_img)
            sub_matrix = self.construst_sub_matrix(feature_3d)
            r_w = self.construst_R_weights(sub_matrix)
            targets_1['r_w_s'+ str(i)] = torch.from_numpy(r_w).float()
            rgb_img = rgb_img[::2,::2,:]

        final_img = torch.from_numpy(np.ascontiguousarray(np.transpose(srgb_img, (2,0,1)))).contiguous().float()

        sparse_shading_name = str(self.height) + "x" + str(self.width)

        if self.current_o_idx == 0:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 1:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name +  "/R0.h5"
        elif self.current_o_idx == 2:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 3:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"
        elif self.current_o_idx == 4:
            sparse_path_1s = self.root + "/CGIntrinsics/IIW/sparse_hdf5_S/" + sparse_shading_name + "/R0.h5"

        return final_img, targets_1, sparse_path_1s


    def __len__(self):
        return len(self.img_list[self.current_o_idx])
