from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
from pytorch_toolbelt import losses as L


def SoftCrossEntropy_DiceLoss(smooth_factor=0.0, w_ce=0.9, w_dice=0.1):
    # combine soft celoss with diceloss
    softce_fn = SoftCrossEntropyLoss(smooth_factor=smooth_factor)
    diceloss_fn = DiceLoss(mode='multiclass')
    combined_loss = L.JointLoss(first=diceloss_fn, second=softce_fn, first_weight=w_dice, second_weight=w_ce)
    return combined_loss