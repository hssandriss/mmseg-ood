from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log
import numpy as np


@HOOKS.register_module()
class ParseEpochToDecodeHead(Hook):
    def before_run(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            runner.model.module.decode_head.total_epochs = runner._max_epochs

    def before_train_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            runner.model.module.decode_head.epoch_num = runner.epoch

    def after_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            if runner.model.module.decode_head.epoch_num != runner.epoch:
                print_log(f"model is at epoch: {runner.model.module.decode_head.epoch_num}, runner is at epoch: {runner.epoch}")
                
        if hasattr(runner.model.module.decode_head, "kl_vals"):
            assert all(runner.model.module.decode_head.kl_weights[0] == w for w in runner.model.module.decode_head.kl_weights[1:])
            print_log(f"KL weight: {float(np.mean(runner.model.module.decode_head.kl_weights)):.2f}")
            print_log(f"Avg Epoch KL term: {float(np.mean(runner.model.module.decode_head.kl_vals)):.2f}")
            runner.model.module.decode_head.kl_weights = []
            runner.model.module.decode_head.kl_vals = []

    def before_train_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            if runner.epoch > 0 and (
                    runner.model.module.decode_head.epoch_num + 1 != runner.epoch or
                    runner.model.module.decode_head.total_epochs != runner._max_epochs):
            runner.model.module.decode_head.epoch_num = runner.epoch
            runner.model.module.decode_head.total_epochs = runner._max_epochs
