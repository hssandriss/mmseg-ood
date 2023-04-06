# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log


@HOOKS.register_module()
class ParseEpochToDecodeHead(Hook):

    def before_run(self, runner):
        model = runner.model.module
        if hasattr(model.decode_head, 'epoch_num'):
            runner.model.module.decode_head.total_epochs = runner._max_epochs

    def after_epoch(self, runner):
        model = runner.model.module
        if hasattr(model.decode_head, 'epoch_num'):
            if model.decode_head.epoch_num != runner.epoch:
                print_log(f'model is at epoch: {model.decode_head.epoch_num},'
                          ' runner is at epoch: {runner.epoch}')

        if hasattr(model.decode_head, 'kl_vals'):
            model = runner.model.module
            assert all(model.decode_head.kl_weights[0] == w
                       for w in model.decode_head.kl_weights[1:])
            print_log('KL weight: '
                      f'{float(np.mean(model.decode_head.kl_weights)):.2f}')
            print_log('Avg Epoch KL term: '
                      f'{float(np.mean(model.decode_head.kl_vals)):.2f}')
            model.decode_head.kl_weights = []
            model.decode_head.kl_vals = []

    def before_train_epoch(self, runner):
        model = runner.model.module
        if hasattr(model.decode_head, 'epoch_num'):
            if runner.epoch > 0 and (
                    model.decode_head.epoch_num + 1 != runner.epoch
                    or model.decode_head.total_epochs != runner._max_epochs):
                model.decode_head.epoch_num = runner.epoch
                model.decode_head.total_epochs = runner._max_epochs
