import os
import sys
import torch
# import tensorflow as tf
import h5py
# import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.utils import *
from torchvision import transforms as T
from functools import partial
from pathlib import Path
from torch import nn, einsum
from PIL import Image
import cv2

def exists(x):
    return x is not None



class MRI_PET_Dataset(Dataset):
    def __init__(
        self,
        config,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.config = config
        self.folder_mri = os.path.join(folder, 'MRI_png')
        self.folder_pet = os.path.join(folder, 'PET_png')
        self.image_size = image_size
        self.paths_mri = [p for ext in exts for p in Path(f'{self.folder_mri}').glob(f'**/*.{ext}')]
        self.paths_pet = [q for ext in exts for q in Path(f'{self.folder_pet}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=1) if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths_mri)

    def __getitem__(self, index):
        path_mri = self.paths_mri[index]
        path_pet = self.paths_pet[index]
        file_name_mri = os.path.basename(path_mri)
        file_name_pet = os.path.basename(path_pet)
        print('file_name_mri:', file_name_mri)
        print('file_name_pet:',file_name_pet)

        img_mri = Image.open(path_mri).convert("L") 
        img_mri = self.transform(img_mri)
        img_pet = Image.open(path_pet).convert("L")
        img_pet = self.transform(img_pet)
        if self.config.data.normalize_type == 'minmax':
            img_mri = normalize(img_mri)
            img_pet = normalize(img_pet)
        elif self.config.data.normalize_type == 'std':
            # mri
            img_mri = np.expand_dims(img_mri, 0)
            ksp_mri_idx = FFT2c(img_mri)
            minv_mri = np.std(ksp_mri_idx)
            ksp_mri_idx = ksp_mri_idx / (self.config.data.normalize_coeff * minv_mri)
            img_mri = IFFT2c(ksp_mri_idx)
            img_mri = np.squeeze(img_mri, 0)
            img_mri = img_mri.real.astype(np.float32)
            # pet
            img_pet = np.expand_dims(img_pet, 0)
            ksp_pet_idx = FFT2c(img_pet)
            minv_pet = np.std(ksp_pet_idx)
            ksp_pet_idx = ksp_pet_idx / (self.config.data.normalize_coeff * minv_pet)
            img_pet = IFFT2c(ksp_pet_idx)
            img_pet = np.squeeze(img_pet, 0)
            img_pet = img_pet.real.astype(np.float32)

        return img_pet, img_mri, file_name_mri




def get_dataset(config, mode):
    print("Dataset name:", config.data.dataset_name)
    if config.data.dataset_name == 'pet':
        dataset = PET_Dataset(config, 
                    ' ',
                    image_size=128,
                    augment_horizontal_flip=True)

    elif config.data.dataset_name == 'mripet':
        if mode == 'training':
            dataset = MRI_PET_Dataset(config, 
                        ' ', image_size=128,
                        augment_horizontal_flip=False)
        else:
            dataset = MRI_PET_Dataset(config, 
                        ' ', image_size=128,
                        augment_horizontal_flip=False)
    else:
        dataset = FastMRIKneeDataSet(config, mode)
    
    print('dataset:', config.data.dataset_name)

    if mode == 'training':
        data = DataLoader(
            dataset, batch_size=config.training.batch_size, shuffle=True, pin_memory=True)
    else:
        data = DataLoader(
            dataset, batch_size=config.sampling.batch_size, shuffle=False, pin_memory=True)

    print(mode, "data loaded")

    return data

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, psnr_input, psnr_output):
        mse = self.mse(psnr_output, psnr_input) + 1e-12
        return -10 * torch.log(mse) / np.log(10)

def rotate(data, ori = 'left'):  

    if ori == 'left':
        data = list(map(list,zip(*data)))[:-1]
    else:
        data = list(map(list,zip(*data[:-1])))
            
    data = np.array(data)
    return data

'''
rotate 90 degree
'''
def get_rotate_90_image(picture):
    rotate_img = np.copy(picture)
    for i in range(picture.shape[0]):
        for j in range(picture.shape[1]):
            new_x = 127 - j 
            new_j =  i
            rotate_img[new_x, new_j] = picture[i, j]
            
    return rotate_img


