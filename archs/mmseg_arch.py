import os
import torch
import torch.nn as nn
# make sure install mmcv-full and mmsegmentation with compatible version
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor

class MMSegArch(nn.Module):
    """
        Use the smp library to construct arch
        simple modify in this class if necessary 
    """
    def __init__(self, arch_config='fcn_hr18', classes=3, checkpoint_file=None, device='cuda'):
        super(MMSegArch, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'mmseg_arch_configs', f'{arch_config}.py')
        config = mmcv.Config.fromfile(config_path)
        config['decode_head']['num_classes'] = classes

        self.arch = init_segmentor(config, checkpoint_file, device=device)

    def forward(self, x):
        feat_ls = self.arch.extract_feat(x)
        seg_logits = self.arch.decode_head(feat_ls)
        return seg_logits

if __name__ == "__main__":
    
    arch = MMSegArch(arch_config='fcn_hr18', device='cuda')
    dummy_input = torch.rand(1, 3, 64, 64).cuda()
    arch.eval()
    output = arch(dummy_input)
    print(output.size())