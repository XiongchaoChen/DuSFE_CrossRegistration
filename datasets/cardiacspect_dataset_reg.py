import os
import h5py
import random
import numpy as np
import pdb
import torch
import torchvision.utils as utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.data_patch_util import *
from utils.function import *

def nmae(sr_image, gt_image):
    diff_abs = np.absolute(sr_image - gt_image)
    mae = np.mean(diff_abs)
    nmae_ = mae/np.mean(gt_image)
    return nmae_


# ----------------------- Training Dataset ---------------------
class CardiacSPECT_Train_Reg(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_train
        self.n_patch = opts.n_patch_train
        # self.AUG = opts.AUG  # No augmentation here

        self.data_dir = os.path.join(self.root, 'train')
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_Amap_Trans_all = []
        self.vol_Amap_CT_all = []
        self.vol_SPECT_NC_all = []
        self.vol_SPECT_SC_all = []
        self.vol_Index_Trans_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_Amap_Trans = f['Amap_Trans'][...].transpose(2,1,0)  
                vol_Amap_CT = f['Amap_CT'][...].transpose(2,1,0)  
                vol_SPECT_NC = f['SPECT_NC'][...].transpose(2,1,0)  
                vol_SPECT_SC = f['SPECT_SC'][...].transpose(2,1,0)  
                vol_Index_Trans = f['Index_Trans'][...]  

            # create the random index for cropping patches
            X_template = vol_Amap_Trans
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_Amap_Trans, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_Amap_Trans_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_Amap_CT, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_Amap_CT_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SPECT_NC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_SPECT_NC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SPECT_SC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_SPECT_SC_all.append(X_patches)

            # Indexes
            X_patches = vol_Index_Trans.transpose(1, 0)  
            X_patches = X_patches.repeat(self.n_patch, axis=0) 
            self.vol_Index_Trans_all.append(X_patches)

        self.vol_Amap_Trans_all = np.concatenate(self.vol_Amap_Trans_all, 0)  
        self.vol_Amap_CT_all = np.concatenate(self.vol_Amap_CT_all, 0)
        self.vol_SPECT_NC_all = np.concatenate(self.vol_SPECT_NC_all, 0)
        self.vol_SPECT_SC_all = np.concatenate(self.vol_SPECT_SC_all, 0)
        self.vol_Index_Trans_all = np.concatenate(self.vol_Index_Trans_all, 0) 

    def __getitem__(self, index):
        vol_Amap_Trans = self.vol_Amap_Trans_all[index, ...]  
        vol_Amap_CT = self.vol_Amap_CT_all[index, ...]
        vol_SPECT_NC = self.vol_SPECT_NC_all[index, ...]
        vol_SPECT_SC = self.vol_SPECT_SC_all[index, ...]
        vol_Index_Trans = self.vol_Index_Trans_all[index, ...] 

        vol_Amap_Trans = torch.from_numpy(vol_Amap_Trans.copy())
        vol_Amap_CT = torch.from_numpy(vol_Amap_CT.copy())
        vol_SPECT_NC = torch.from_numpy(vol_SPECT_NC.copy())
        vol_SPECT_SC = torch.from_numpy(vol_SPECT_SC.copy())
        vol_Index_Trans = torch.from_numpy(vol_Index_Trans.copy())

        return {'vol_Amap_Trans': vol_Amap_Trans,
                'vol_Amap_CT': vol_Amap_CT,
                'vol_SPECT_NC': vol_SPECT_NC,
                'vol_SPECT_SC': vol_SPECT_SC,
                'vol_Index_Trans': vol_Index_Trans}

    def __len__(self):
        return self.vol_Amap_Trans_all.shape[0]  



# ----------------------- Testing Dataset ---------------------
class CardiacSPECT_Test_Reg(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_test
        self.n_patch = opts.n_patch_test

        self.data_dir = os.path.join(self.root, 'test')  
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_Amap_Trans_all = []
        self.vol_Amap_CT_all = []
        self.vol_SPECT_NC_all = []
        self.vol_SPECT_SC_all = []
        self.vol_Index_Trans_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_Amap_Trans = f['Amap_Trans'][...].transpose(2, 1, 0)  
                vol_Amap_CT = f['Amap_CT'][...].transpose(2, 1, 0)  
                vol_SPECT_NC = f['SPECT_NC'][...].transpose(2, 1, 0) 
                vol_SPECT_SC = f['SPECT_SC'][...].transpose(2, 1, 0)  
                vol_Index_Trans = f['Index_Trans'][...]  

            # create the random index for cropping patches
            X_template = vol_Amap_Trans
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_Amap_Trans, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_Amap_Trans_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_Amap_CT, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_Amap_CT_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SPECT_NC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :] 
            self.vol_SPECT_NC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SPECT_SC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_SPECT_SC_all.append(X_patches)

            # Indexes
            X_patches = vol_Index_Trans.transpose(1, 0)  
            X_patches = X_patches.repeat(self.n_patch, axis=0)  
            self.vol_Index_Trans_all.append(X_patches)

        self.vol_Amap_Trans_all = np.concatenate(self.vol_Amap_Trans_all, 0) 
        self.vol_Amap_CT_all = np.concatenate(self.vol_Amap_CT_all, 0)
        self.vol_SPECT_NC_all = np.concatenate(self.vol_SPECT_NC_all, 0)
        self.vol_SPECT_SC_all = np.concatenate(self.vol_SPECT_SC_all, 0)
        self.vol_Index_Trans_all = np.concatenate(self.vol_Index_Trans_all, 0)  

    def __getitem__(self, index):
        vol_Amap_Trans = self.vol_Amap_Trans_all[index, ...]  
        vol_Amap_CT = self.vol_Amap_CT_all[index, ...]
        vol_SPECT_NC = self.vol_SPECT_NC_all[index, ...]
        vol_SPECT_SC = self.vol_SPECT_SC_all[index, ...]
        vol_Index_Trans = self.vol_Index_Trans_all[index, ...]  

        vol_Amap_Trans = torch.from_numpy(vol_Amap_Trans.copy())
        vol_Amap_CT = torch.from_numpy(vol_Amap_CT.copy())
        vol_SPECT_NC = torch.from_numpy(vol_SPECT_NC.copy())
        vol_SPECT_SC = torch.from_numpy(vol_SPECT_SC.copy())
        vol_Index_Trans = torch.from_numpy(vol_Index_Trans.copy())

        return {'vol_Amap_Trans': vol_Amap_Trans,
                'vol_Amap_CT': vol_Amap_CT,
                'vol_SPECT_NC': vol_SPECT_NC,
                'vol_SPECT_SC': vol_SPECT_SC,
                'vol_Index_Trans': vol_Index_Trans}

    def __len__(self):
        return self.vol_Amap_Trans_all.shape[0]  



# ----------------------- Validation Dataset ---------------------
class CardiacSPECT_Valid_Reg(Dataset):
    def __init__(self, opts=None):
        self.root = opts.data_root
        self.patch_size = opts.patch_size_valid
        self.n_patch = opts.n_patch_valid

        self.data_dir = os.path.join(self.root, 'valid')  
        self.data_files = sorted([os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.h5')])

        self.vol_Amap_Trans_all = []
        self.vol_Amap_CT_all = []
        self.vol_SPECT_NC_all = []
        self.vol_SPECT_SC_all = []
        self.vol_Index_Trans_all = []

        # load all images and patching
        for filename in self.data_files:
            print('Patching: ' + str(filename))

            with h5py.File(filename, 'r') as f:
                vol_Amap_Trans = f['Amap_Trans'][...].transpose(2, 1, 0) 
                vol_Amap_CT = f['Amap_CT'][...].transpose(2, 1, 0)  
                vol_SPECT_NC = f['SPECT_NC'][...].transpose(2, 1, 0)  
                vol_SPECT_SC = f['SPECT_SC'][...].transpose(2, 1, 0)  
                vol_Index_Trans = f['Index_Trans'][...]  

            # create the random index for cropping patches
            X_template = vol_Amap_Trans
            indexes = get_random_patch_indexes(data=X_template, patch_size=self.patch_size, num_patches=self.n_patch, padding='VALID')

            # use index to crop patches
            X_patches = get_patches_from_indexes(image=vol_Amap_Trans, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_Amap_Trans_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_Amap_CT, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :] 
            self.vol_Amap_CT_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SPECT_NC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_SPECT_NC_all.append(X_patches)

            X_patches = get_patches_from_indexes(image=vol_SPECT_SC, indexes=indexes, patch_size=self.patch_size, padding='VALID', dtype=None)
            X_patches = X_patches[:, np.newaxis, :, :, :]  
            self.vol_SPECT_SC_all.append(X_patches)

            # Indexes
            X_patches = vol_Index_Trans.transpose(1, 0)  
            X_patches = X_patches.repeat(self.n_patch, axis=0)  
            self.vol_Index_Trans_all.append(X_patches)

        self.vol_Amap_Trans_all = np.concatenate(self.vol_Amap_Trans_all, 0)  
        self.vol_Amap_CT_all = np.concatenate(self.vol_Amap_CT_all, 0)
        self.vol_SPECT_NC_all = np.concatenate(self.vol_SPECT_NC_all, 0)
        self.vol_SPECT_SC_all = np.concatenate(self.vol_SPECT_SC_all, 0)
        self.vol_Index_Trans_all = np.concatenate(self.vol_Index_Trans_all, 0)  

    def __getitem__(self, index):
        vol_Amap_Trans = self.vol_Amap_Trans_all[index, ...]  
        vol_Amap_CT = self.vol_Amap_CT_all[index, ...]
        vol_SPECT_NC = self.vol_SPECT_NC_all[index, ...]
        vol_SPECT_SC = self.vol_SPECT_SC_all[index, ...]
        vol_Index_Trans = self.vol_Index_Trans_all[index, ...]  

        vol_Amap_Trans = torch.from_numpy(vol_Amap_Trans.copy())
        vol_Amap_CT = torch.from_numpy(vol_Amap_CT.copy())
        vol_SPECT_NC = torch.from_numpy(vol_SPECT_NC.copy())
        vol_SPECT_SC = torch.from_numpy(vol_SPECT_SC.copy())
        vol_Index_Trans = torch.from_numpy(vol_Index_Trans.copy())

        return {'vol_Amap_Trans': vol_Amap_Trans,
                'vol_Amap_CT': vol_Amap_CT,
                'vol_SPECT_NC': vol_SPECT_NC,
                'vol_SPECT_SC': vol_SPECT_SC,
                'vol_Index_Trans': vol_Index_Trans}

    def __len__(self):
        return self.vol_Amap_Trans_all.shape[0] 

if __name__ == '__main__':
    aa = 1
