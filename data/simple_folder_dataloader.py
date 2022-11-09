import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from glob import glob


class SimpleFolderDataset(Dataset):
    """
        Images and labels are arranged in different folders, 
        with image name for correspondence 
        (location after sorted should be consistent):
            dataroot/
                imgs/
                    000.jpg
                    001.jpg
                lbls/
                    000_lbl.png
                    001_lbl.png
                ...
        Returns:
            {"img": image, "label": label}
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot_img = opt_dataset['dataroot_img']
        self.dataroot_lbl = opt_dataset['dataroot_lbl']
        self.phase = opt_dataset['phase']
        self.img_exts = opt_dataset['img_exts']  # list, 'cause input may be different formats
        self.lbl_exts = opt_dataset['lbl_exts']

        augment_opt = opt_dataset['augment']
        augment_type = augment_opt.pop('augment_type')
        if self.phase == 'train':
            self.augment = importlib.import_module(f'data_augment.{augment_type}').train_augment(**augment_opt)
        elif self.phase == 'valid':
            self.augment = importlib.import_module(f'data_augment.{augment_type}').val_augment(**augment_opt)

        # collect all images and labels recursively under given folders
        img_paths = list()
        for img_ext in self.img_exts:
            img_paths += list(glob(os.path.join(self.dataroot_img, f'*.{img_ext}')))
        self.img_paths = sorted(img_paths)

        lbl_paths = list()
        for lbl_ext in self.lbl_exts:
            lbl_paths += list(glob(os.path.join(self.dataroot_lbl, f'*.{lbl_ext}')))
        self.lbl_paths = sorted(lbl_paths)


    def __getitem__(self, index):

        cur_img_path = self.img_paths[index]
        cur_lbl_path = self.lbl_paths[index]
        cur_img = cv2.imread(cur_img_path)
        # TODO: currently only support 1-channel int map
        # should support color label map handling also
        cur_lbl = cv2.imread(cur_lbl_path, cv2.IMREAD_UNCHANGED)

        img_lbl_aug = self.augment(image=cur_img, mask=cur_lbl)
        img_aug, lbl_aug = img_lbl_aug['image'], img_lbl_aug['mask'].to(torch.int64)
        # simple dataset cannot cover mixup/contrast etc. which need 2 or more images to return
        output_dict = {
            "img" : img_aug,
            "label": lbl_aug,
            "img_path": cur_img_path,
            "lbl_path": cur_lbl_path
        }
        return output_dict

    def __len__(self):
        return len(self.img_paths)


def SimpleFolderDataloader(opt_dataloader):
    phase = opt_dataloader['phase']
    if phase == 'train':
        batch_size = opt_dataloader['batch_size']
        num_workers = opt_dataloader['num_workers']
        shuffle = True
    elif phase == 'valid':
        batch_size = 1
        num_workers = 0
        shuffle = False
    folder_dataset = SimpleFolderDataset(opt_dataloader)
    dataloader = DataLoader(folder_dataset, batch_size=batch_size, pin_memory=True, \
                            drop_last=True, shuffle=shuffle, num_workers=num_workers)
    return dataloader