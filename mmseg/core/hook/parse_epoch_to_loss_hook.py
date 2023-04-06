# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log


@HOOKS.register_module()
class ParseEpochToLossHook(Hook):

    def before_run(self, runner):
        model = runner.model.module
        if hasattr(model.decode_head.loss_decode, 'epoch_num'):
            model.decode_head.loss_decode.total_epochs = runner._max_epochs

    def before_train_epoch(self, runner):
        model = runner.model.module
        if hasattr(model.decode_head.loss_decode, 'epoch_num'):
            model.decode_head.loss_decode.epoch_num = runner.epoch

    def after_epoch(self, runner):
        model = runner.model.module
        if hasattr(model.decode_head.loss_decode, 'epoch_num'):
            if model.decode_head.loss_decode.epoch_num != runner.epoch:
                print_log('Descrepancy in the stored epoch number " \
                    "between model and runner')
