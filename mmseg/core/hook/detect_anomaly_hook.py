import torch
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log


@HOOKS.register_module()
class DetectAnomalyHook(Hook):

    def before_train_epoch(self, runner):
        torch.autograd.set_detect_anomaly(True)

    def after_train_epoch(self, runner):
        torch.autograd.set_detect_anomaly(False)
