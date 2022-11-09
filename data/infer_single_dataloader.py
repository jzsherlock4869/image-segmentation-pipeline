import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from glob import glob


class InferSingleDataset(Dataset):
    """
    Images for inference in single folder
        test_images/
            test_img1.png
            test_img2.jpg
            ...
        Returns:
            {"img": image, "img_path": impath, "ori_size_wh": ori_size}
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot_img = opt_dataset['dataroot_img']

        augment_opt = opt_dataset['augment']
        augment_type = augment_opt.pop('augment_type')
        self.augment = importlib.import_module(f'data_augment.{augment_type}').val_augment(**augment_opt)

        # collect all images under given folder
        self.img_paths = sorted(list(glob(os.path.join(self.dataroot_img, '*'))))

    def __getitem__(self, index):

        cur_img_path = self.img_paths[index]
        cur_img = cv2.imread(cur_img_path)
        ori_size_wh = (cur_img.shape[1], cur_img.shape[0])

        img_aug = self.augment(image=cur_img)['image']
        # simple dataset cannot cover mixup/contrast etc. which need 2 or more images to return
        output_dict = {
            "img": img_aug,
            "img_path": cur_img_path,
            "ori_size_wh": ori_size_wh,
        }
        return output_dict

    def __len__(self):
        return len(self.img_paths)


def InferSingleDataloader(opt_dataloader):
    folder_dataset = InferSingleDataset(opt_dataloader)
    dataloader = DataLoader(folder_dataset, batch_size=1, pin_memory=True, \
                            drop_last=True, shuffle=False, num_workers=0)
    return dataloader