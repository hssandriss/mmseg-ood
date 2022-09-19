from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log


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

    def before_train_epoch(self, runner):
        if hasattr(runner.model.module.decode_head, "epoch_num"):
            if runner.epoch > 0 and (
                    runner.model.module.decode_head.epoch_num + 1 != runner.epoch or
                    runner.model.module.decode_head.total_epochs != runner._max_epochs):
                import ipdb; ipdb.set_trace()
            runner.model.module.decode_head.epoch_num = runner.epoch
            runner.model.module.decode_head.total_epochs = runner._max_epochs
