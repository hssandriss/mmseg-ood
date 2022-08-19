from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log


@HOOKS.register_module()
class ParseEpochToLossHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def before_train_epoch(self, runner):
        print_log(f"Epoch ---> {runner.epoch}/{runner._max_epochs}")
        if hasattr(runner.model.module.decode_head.loss_decode, "epoch_num"):
            if runner.epoch > 0 and (
                    runner.model.module.decode_head.loss_decode.epoch_num + 1 != runner.epoch or runner.model.module.decode_head.loss_decode.total_epochs !=
                    runner._max_epochs):
                import ipdb; ipdb.set_trace()
            runner.model.module.decode_head.loss_decode.epoch_num = runner.epoch
            runner.model.module.decode_head.loss_decode.total_epochs = runner._max_epochs
