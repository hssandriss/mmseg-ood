# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class DetectAnomalyHook(Hook):

    def before_train_iter(self, runner):
        torch.autograd.set_detect_anomaly(True)

    def after_train_iter(self, runner):
        torch.autograd.set_detect_anomaly(False)

    def before_iter(self, runner):
        torch.cuda.empty_cache()
