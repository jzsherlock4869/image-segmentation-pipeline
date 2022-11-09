import torch
import numpy as np


class SegMetricAll:
    """
        segmentation metrics contains IoU / FWIoU / Acc
        use metric_names to select required metircs
    """
    def __init__(self, num_classes, metric_type='miou') -> None:
        self.num_classes = num_classes
        self.conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.metric_type = metric_type
        self.sample_count = 0
        assert metric_type in ['miou', 'fwiou', 'acc']

    def update(self, lbl, pred):
        # pred and label should be torch.Tensor with range [0, num_classes)
        bs = lbl.size()[0]
        self.sample_count += bs
        pred = pred.detach().cpu().numpy().astype(np.int64)
        lbl = lbl.detach().cpu().numpy().astype(np.int64)
        mask = (lbl >= 0) & (lbl < self.num_classes)
        label = self.num_classes * lbl[mask] + pred[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        tmp_conf_mat = count.reshape(self.num_classes, self.num_classes)
        self.conf_mat += tmp_conf_mat

    def num_sample(self):
        return self.sample_count

    def reset(self):
        self.sample_count = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def _calc_miou(self):
        ious = np.diag(self.conf_mat) / (
            np.sum(self.conf_mat, axis=1) +
            np.sum(self.conf_mat, axis=0) -
            np.diag(self.conf_mat)
        )
        miou = np.nanmean(ious)
        return {'miou': miou, 'ious': ious}

    def _calc_fwiou(self):
        # row same real, column same pred
        freq = np.sum(self.conf_mat, axis=1) / np.sum(self.conf_mat)
        ious = np.diag(self.conf_mat) / (
            np.sum(self.conf_mat, axis=1) +
            np.sum(self.conf_mat, axis=0) - 
            np.diag(self.conf_mat)
        )
        fwiou = np.sum(freq[freq > 0] * ious[freq > 0])
        return {'fwiou': fwiou, 'ious': ious, 'freq': freq}

    def _calc_acc(self):
        # overall acc
        overall_acc = np.sum(np.diag(self.conf_mat)) / np.sum(self.conf_mat)
        return overall_acc

    def calc(self):
        if self.metric_type.lower() == 'miou':
            ret_val = self._calc_miou()
        elif self.metric_type.lower() == 'fwiou':
            ret_val = self._calc_fwiou()
        elif self.metric_type.lower() == 'acc':
            ret_val = self._calc_acc()
        else:
            raise NotImplementedError(f'metric type {self.metric_type} unrecognized')
        return ret_val

    def calc_confusion_mat(self):
        return self.conf_mat
