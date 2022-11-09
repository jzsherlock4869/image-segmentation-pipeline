import os
import os.path as osp
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../datasets/weizmann_horse_db/rgb'
lbl_dir = '../datasets/weizmann_horse_db/figure_ground'
output_dir = '../datasets/weizmann_horse_split'
val_ratio = 0.2

os.makedirs(osp.join(output_dir, 'train_split/images'), exist_ok=True)
os.makedirs(osp.join(output_dir, 'train_split/masks'), exist_ok=True)
os.makedirs(osp.join(output_dir, 'valid_split/images'), exist_ok=True)
os.makedirs(osp.join(output_dir, 'valid_split/masks'), exist_ok=True)

imnames = sorted(os.listdir(img_dir))
lblnames = sorted(os.listdir(lbl_dir))
assert len(imnames) == len(lblnames)
train_ids, val_ids = train_test_split(range(len(imnames)), test_size=val_ratio, random_state=42)
print(len(train_ids), len(val_ids))

for i in range(len(imnames)):
    impath = osp.join(img_dir, imnames[i])
    lblpath = osp.join(lbl_dir, lblnames[i])
    lbl_wb = cv2.imread(lblpath)[:,:,0]
    mask = (lbl_wb / 255.0).astype(np.uint8)

    if i in train_ids:
        os.system(f"cp {impath} {osp.join(output_dir, 'train_split/images')}")
        cv2.imwrite(osp.join(output_dir, 'train_split/masks', osp.basename(lblpath).split('.')[0] + '.png'), mask)
    else:
        assert i in val_ids
        os.system(f"cp {impath} {osp.join(output_dir, 'valid_split/images')}")
        cv2.imwrite(osp.join(output_dir, 'valid_split/masks', osp.basename(lblpath).split('.')[0] + '.png'), mask)

print('[jzsherlock] all done, everything fine.')