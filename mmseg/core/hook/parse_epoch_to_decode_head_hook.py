from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log
import numpy as np
import torch


@HOOKS.register_module()
class ParseEpochToDecodeHeadHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def before_run(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            runner.model.module.decode_head.total_epochs = runner._max_epochs

    def before_train_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            runner.model.module.decode_head.epoch_num = runner.epoch

    def after_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            if runner.model.module.decode_head.epoch_num != runner.epoch:
                import ipdb; ipdb.set_trace()
        if hasattr(runner.model.module.decode_head, "kl_vals"):
            print_log(f"Avg Epoch KL term: {float(np.mean(runner.model.module.decode_head.kl_vals)):.2f}")
            runner.model.module.decode_head.kl_vals = []

    def before_train_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            if runner.epoch > 0 and (
                    runner.model.module.decode_head.epoch_num + 1 != runner.epoch or
                    runner.model.module.decode_head.total_epochs != runner._max_epochs):
                import ipdb; ipdb.set_trace()
            runner.model.module.decode_head.epoch_num = runner.epoch
            runner.model.module.decode_head.total_epochs = runner._max_epochs

    def before_iter(self, runner):
        torch.autograd.set_detect_anomaly(True)

    def after_iter(self, runner):
        torch.autograd.set_detect_anomaly(False)
