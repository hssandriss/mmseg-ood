# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
import numpy as np
from mmcv.utils import print_log


def exp_schedule(epoch, total_epochs, lo=0.0002, hi=1.):
    return min(max(np.exp(np.log(lo) * epoch / (total_epochs - 1)), lo), hi)


@HOOKS.register_module()
class CustomEMAHook(Hook):
    r"""Exponential Moving Average Hook.

    Use Exponential Moving Average on all parameters of model in training
    process. All parameters have a ema backup, which update by the formula
    as below. EMAHook takes priority over EvalHook and CheckpointSaverHook.

        .. math::

            Xema\_{t+1} = (1 - \text{momentum}) \times
            Xema\_{t} +  \text{momentum} \times X_t

    Args:
        momentum (float): The momentum used for updating ema parameter.
            Defaults to 0.0002.
        interval (int): Update ema parameter every interval iteration.
            Defaults to 1.
        warm_up (int): During first warm_up steps, we may use smaller momentum
            to update ema parameters more slowly. Defaults to 100.
        resume_from (str, optional): The checkpoint path. Defaults to None.
    """

    def __init__(self,
                 momentum: float = 0.0002,
                 interval: int = 1,
                 warm_up: int = 100,
                 warm_up_epochs: int = 0,
                 resume_from: Optional[str] = None):
        assert isinstance(interval, int) and interval > 0
        self.warm_up = warm_up
        self.warm_up_epochs = warm_up_epochs
        self.interval = interval
        assert momentum > 0 and momentum < 1
        self.momentum = momentum**interval
        self.momentum = momentum
        self.checkpoint = resume_from
        self.enabled = False
        # self.schedule = lambda epoch, total_epochs: np.exp(np.log(0.0002) * epoch / (total_epochs - 1))

    def before_run(self, runner):
        """
        To resume model with it's ema parameters more friendly.
        Register ema parameter as ``named_buffer`` to model
        """
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        self.param_ema_buffer = {}
        # self.model_parameters is a reference to model params and not a copy
        # dict(runner.model.module.named_parameters(recurse=True))['backbone.stem.0.weight'] is self.model_parameters['backbone.stem.0.weight']
        self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            # "." is not allowed in module's buffer name
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))
        if self.checkpoint is not None:
            runner.resume(self.checkpoint)

    def after_train_iter(self, runner):
        """
        Update ema parameter every self.interval iterations.
        Here Buffer Parameters are updated no model parameters
        """
        curr_step = runner.iter
        # We warm up the momentum considering the instability at beginning
        momentum = min(self.momentum, (1 + curr_step) / (self.warm_up + curr_step))
        # momentum = self.momentum
        if curr_step % self.interval != 0 and runner.epoch > self.warm_up_epochs:
            return
        self.enabled = True
        for name, parameter in self.model_parameters.items():
            buffer_name = self.param_ema_buffer[name]
            buffer_parameter = self.model_buffers[buffer_name]
            # current= (1-m)*prev + m*current
            buffer_parameter.mul_(1 - momentum).add_(momentum * parameter.data)

    def after_train_epoch(self, runner):
        """
        We load parameter values from ema backup to model before the EvalHook.
        """
        if self.enabled:
            self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        """
        We recover model's parameter from ema backup after last epoch's EvalHook.
        """
        if self.enabled:
            self._swap_ema_parameters()
            # self.momentum = exp_schedule(runner.epoch, runner._max_epochs, lo=0.0002, hi=1.)
            print_log(f"EMA momentum: {self.momentum}")

    def _swap_ema_parameters(self):
        """Swap the parameter of model with parameter in ema_buffer."""
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)
