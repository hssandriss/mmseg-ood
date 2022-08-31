from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import print_log


@HOOKS.register_module()
class ParseEpochToLossHook(Hook):

    def before_run(self, runner):
        if hasattr(runner.model.module.decode_head.loss_decode, "epoch_num"):
            runner.model.module.decode_head.loss_decode.total_epochs = runner._max_epochs

    def before_train_epoch(self, runner):
        # print_log(f"Epoch ---> {runner.epoch}/{runner._max_epochs}")
        if hasattr(runner.model.module.decode_head.loss_decode, "epoch_num"):
            runner.model.module.decode_head.loss_decode.epoch_num = runner.epoch

    def after_epoch(self, runner):
        if runner.model.module.decode_head.loss_decode.epoch_num != runner.epoch:
            import ipdb; ipdb.set_trace()
