import random
import numpy as np
import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.image_folder import *
from relighting_image_folder import RelightingImageFolder
import scipy.io as sio
from builtins import object
import sys
import h5py

from torch.utils.data.dataloader import default_collate, container_abcs, string_classes, int_classes

NUM_WORKERS = 8


def my_collate(batch, field_skip = ['eq_mat','ineq_mat', 'gt_eq_mat', 'gt_ineq_mat']):
    # Treat the data in field_skip fields as special (don't concat, just keep the tensors as a list).  This lets us handle eq_mat and ineq_mat in the data loader which should allow for a speedup.
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    #print elem_type
    if isinstance(batch[0], torch.Tensor):
        return default_collate(batch)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return default_collate(batch)
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) if not (key in field_skip) else [d[key] for d in batch] for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]
    else:
        default_collate(batch)

class RelightingTestData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        return self

    def __next__(self):
        return next(self.data_loader_iter)

class IIWTestData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        final_img, target_1, sparse_path_1s  = next(self.data_loader_iter)
        return {'img_1': final_img, 'target_1': target_1}

class SAWData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def sparse_loader(self, sparse_path, num_features):
        # print("sparse_path  ", sparse_path)
        # sys.exit()
        hdf5_file_sparse = h5py.File(sparse_path,'r')
        B_arr = []
        data_whole = hdf5_file_sparse.get('/sparse/mn')
        mn = np.array(data_whole)
        mn = np.transpose(mn, (1,0))
        m = int(mn[0][0])
        n = int(mn[1][0])
        # print(m, n)
        data_whole = hdf5_file_sparse.get('/sparse/S')
        S_coo = np.array(data_whole)
        S_coo = np.transpose(S_coo, (1,0))
        S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

        # print(S_coo[:,0:2])
        # print(torch.FloatTensor([3, 4]))
        S_i = S_coo[0:2,:].long()
        S_v = S_coo[2,:].float()
        S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        for i in range(num_features+1):
            data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
            B_coo = np.array(data_whole)
            B_coo = np.transpose(B_coo, (1,0))
            B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
            B_i = B_coo[0:2,:].long()
            B_v = B_coo[2,:].float()

            B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_arr.append(B_mat)


        data_whole = hdf5_file_sparse.get('/sparse/N')
        N = np.array(data_whole)
        N = np.transpose(N, (1,0))
        N = torch.from_numpy(N)

        hdf5_file_sparse.close()
        return S, B_arr, N


    def __next__(self):
        self.iter += 1
        final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

        target_1['SS'] = []
        target_1['SB_list'] = []
        target_1['SN'] = []

        SS_1, SB_list_1, SN_1  = self.sparse_loader(sparse_path_1s[0], 2)

        for i in range(len(sparse_path_1s)):
            target_1['SS'].append(SS_1)
            target_1['SB_list'].append(SB_list_1)
            target_1['SN'].append(SN_1)

        return {'img_1': final_img, 'target_1': target_1}



class CGIntrinsicsData(object):
    def __init__(self, data_loader, root):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.root = root
        # st()
        self.npixels = (256 * 256* 29)
        self.sparse = None
        self.sparse_path = ''

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def sparse_loader(self, sparse_path, num_features):
        # print("sparse_path  ", sparse_path)
        # sys.exit()
        hdf5_file_sparse = h5py.File(sparse_path,'r')
        B_arr = []
        data_whole = hdf5_file_sparse.get('/sparse/mn')
        mn = np.array(data_whole)
        mn = np.transpose(mn, (1,0))
        m = int(mn[0][0])
        n = int(mn[1][0])
        # print(m, n)
        data_whole = hdf5_file_sparse.get('/sparse/S')
        S_coo = np.array(data_whole)
        S_coo = np.transpose(S_coo, (1,0))
        S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

        # print(S_coo[:,0:2])
        # print(torch.FloatTensor([3, 4]))
        S_i = S_coo[0:2,:].long()
        S_v = S_coo[2,:].float()
        S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        for i in range(num_features+1):
            data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
            B_coo = np.array(data_whole)
            B_coo = np.transpose(B_coo, (1,0))
            B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
            B_i = B_coo[0:2,:].long()
            B_v = B_coo[2,:].float()

            B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_arr.append(B_mat)


        data_whole = hdf5_file_sparse.get('/sparse/N')
        N = np.array(data_whole)
        N = np.transpose(N, (1,0))
        N = torch.from_numpy(N)

        hdf5_file_sparse.close()
        return S, B_arr, N

    def __next__(self):
        self.iter += 1
        self.iter += 1
        scale = 4

        final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

        # This is all just precomputed stuff that is the same for every image.
        target_1['SS'] = []
        target_1['SB_list'] = []
        target_1['SN'] = []

        if self.sparse_path != sparse_path_1s[0]:
            self.sparse = self.sparse_loader(sparse_path_1s[0], 2)
            self.sparse_path = sparse_path_1s[0]

        SS_1, SB_list_1, SN_1 = self.sparse

        for i in range(len(sparse_path_1s)):
            target_1['SS'].append(SS_1)
            target_1['SB_list'].append(SB_list_1)
            target_1['SN'].append(SN_1)

        return {'img_1': final_img, 'target_1': target_1}



class IIWData(object):
    def __init__(self, data_loader, flip):
        self.data_loader = data_loader
        # self.fineSize = fineSize
        # self.max_dataset_size = max_dataset_size
        self.flip = flip
        # st()
        self.npixels = (256 * 256* 29)
        self.sparse = None
        self.sparse_path = ''

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def sparse_loader(self, sparse_path, num_features):
        # print("sparse_path  ", sparse_path)
        # sys.exit()
        hdf5_file_sparse = h5py.File(sparse_path,'r')
        B_arr = []
        data_whole = hdf5_file_sparse.get('/sparse/mn')
        mn = np.array(data_whole)
        mn = np.transpose(mn, (1,0))
        m = int(mn[0][0])
        n = int(mn[1][0])
        # print(m, n)
        data_whole = hdf5_file_sparse.get('/sparse/S')
        S_coo = np.array(data_whole)
        S_coo = np.transpose(S_coo, (1,0))
        S_coo = torch.transpose(torch.from_numpy(S_coo),0,1)

        # print(S_coo[:,0:2])
        # print(torch.FloatTensor([3, 4]))
        S_i = S_coo[0:2,:].long()
        S_v = S_coo[2,:].float()
        S = torch.sparse.FloatTensor(S_i, S_v, torch.Size([m+2,n]))

        for i in range(num_features+1):
            data_whole = hdf5_file_sparse.get('/sparse/B'+str(i) )
            B_coo = np.array(data_whole)
            B_coo = np.transpose(B_coo, (1,0))
            B_coo = torch.transpose(torch.from_numpy(B_coo),0,1)
            B_i = B_coo[0:2,:].long()
            B_v = B_coo[2,:].float()

            B_mat = torch.sparse.FloatTensor(B_i, B_v, torch.Size([m+2,m+2]))
            B_arr.append(B_mat)


        data_whole = hdf5_file_sparse.get('/sparse/N')
        N = np.array(data_whole)
        N = np.transpose(N, (1,0))
        N = torch.from_numpy(N)

        hdf5_file_sparse.close()
        return S, B_arr, N


    def __next__(self):
        self.iter += 1
        self.iter += 1
        scale = 4

        final_img, target_1, sparse_path_1s = next(self.data_loader_iter)

        # This is all just precomputed stuff that is the same for every image.
        target_1['SS'] = []
        target_1['SB_list'] = []
        target_1['SN'] = []

        if self.sparse_path != sparse_path_1s[0]:
            self.sparse = self.sparse_loader(sparse_path_1s[0], 2)
            self.sparse_path = sparse_path_1s[0]

        SS_1, SB_list_1, SN_1 = self.sparse

        for i in range(len(sparse_path_1s)):
            target_1['SS'].append(SS_1)
            target_1['SB_list'].append(SB_list_1)
            target_1['SN'].append(SN_1)

        return {'img_1': final_img, 'target_1': target_1}

class CGIntrinsics_TEST_DataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, batch_size):
        transform = None
        dataset = CGIntrinsicsImageFolder(root=_root, \
                list_dir =_list_dir)

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle= False, num_workers=int(NUM_WORKERS), collate_fn=my_collate)
        self.dataset = dataset
        flip = False
        self.paired_data = CGIntrinsicsData(self.data_loader, _root)

    def name(self):
        return 'CGIntrinsics_DataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)

class CGIntrinsics_DataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, batch_size):
        transform = None
        dataset = CGIntrinsicsImageFolder(root=_root, \
                list_dir =_list_dir, clamp_pairs=100)

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle= True, num_workers=int(NUM_WORKERS), collate_fn=my_collate)
        self.dataset = dataset
        flip = False
        self.paired_data = CGIntrinsicsData(self.data_loader, _root)

    def name(self):
        return 'CGIntrinsics_DataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset)



class SAWDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode, batch_size=16):
        # BaseDataLoader.initialize(self)
        # self.fineSize = opt.fineSize

        transform = None

        dataset = SAW_ImageFolder(root=_root, \
                list_dir =_list_dir, mode = mode, is_flip = True, transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle= True, num_workers=int(NUM_WORKERS))

        self.dataset = dataset
        # flip = False
        self.saw_data = SAWData(data_loader)

    def name(self):
        return 'sawDataLoader'

    def load_data(self):
        return self.saw_data

    def __len__(self):
        return len(self.dataset)


class IIWDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode, batch_size=16, is_train=True):
        # is_train lets us use the same dataloader for train and validation
        # BaseDataLoader.initialize(self)
        # self.fineSize = opt.fineSize

        # transformations = [
            # TODO: Scale
            #transforms.CenterCrop((600,800)),
            # transforms.Scale(256, Image.BICUBIC),
            # transforms.ToTensor() ]
        transform = None
        # transform = transforms.Compose(transformations)

        # Dataset A
        # dataset = ImageFolder(root='/phoenix/S6/zl548/AMOS/test/', \
                # list_dir = '/phoenix/S6/zl548/AMOS/test/list/',transform=transform)
        # testset
        dataset = IIW_ImageFolder(root=_root, \
                    list_dir =_list_dir, mode = mode, is_flip = is_train, transform=transform)

        # Have to use 2 or fewer workers for IIW due to having a number of files open during reading.
        data_loader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle = is_train, num_workers=min(int(2), int(NUM_WORKERS)), collate_fn = my_collate)

        self.dataset = dataset
        flip = False
        self.iiw_data = IIWData(data_loader, flip)

    def name(self):
        return 'iiwDataLoader'

    def load_data(self):
        return self.iiw_data

    def __len__(self):
        return len(self.dataset)


class RenderDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, batch_size=16):
        # BaseDataLoader.initialize(self)
        transform = None

        dataset = Render_ImageFolder(root=_root, \
                list_dir =_list_dir, transform=transform)

        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle= True, num_workers=int(NUM_WORKERS))
        self.dataset = dataset

    def name(self):
        return 'renderDataLoader'

    def load_data(self):
        return self.data_loader

    def __len__(self):
        return len(self.dataset)

class RelightingTestDataLoader(BaseDataLoader):
    def __init__(self, _root, batch_size=16):
        dataset = RelightingImageFolder(root=_root, transform=None, loader=None)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = int(NUM_WORKERS))
        self.dataset = dataset
        self.relighting_data = RelightingTestData(data_loader)

    def name(self):
        return 'RelightingTestDataLoader'

    def load_data(self):
        return self.relighting_data

    def __len__(self):
        return len(self.dataset)


class IIWTESTDataLoader(BaseDataLoader):
    def __init__(self,_root, _list_dir, mode, batch_size=16):

        transform = None
        dataset = IIW_ImageFolder(root=_root, \
                list_dir =_list_dir, mode= mode, is_flip = False, transform=transform, load_long_range_annotes=False)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle= False, num_workers=int(NUM_WORKERS), collate_fn = my_collate)
        self.dataset = dataset
        self.iiw_data = IIWTestData(data_loader)

    def name(self):
        return 'IIWTESTDataLoader'

    def load_data(self):
        return self.iiw_data

    def __len__(self):
        return len(self.dataset)

