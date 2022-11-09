import os
import os.path as osp
from sklearn.model_selection import train_test_split

img_dir = '/path/to/all_images'
lbl_dir = '/path/to/all_masks'
output_dir = '/path/to/the/output/folder/for/split/subfolders'
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
    if i in train_ids:
        os.system(f"cp {impath} {osp.join(output_dir, 'train_split/images')}")
        os.system(f"cp {lblpath} {osp.join(output_dir, 'train_split/masks')}")
    else:
        assert i in val_ids
        os.system(f"cp {impath} {osp.join(output_dir, 'valid_split/images')}")
        os.system(f"cp {lblpath} {osp.join(output_dir, 'valid_split/masks')}")

print('[jzsherlock] all done, everything fine.')