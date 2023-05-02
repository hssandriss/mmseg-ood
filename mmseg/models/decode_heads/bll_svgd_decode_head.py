# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.autograd as autograd
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BllSvgdBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BllSvgdBaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        out_channels (int): Output channels of conv_seg.
        threshold (float): Threshold for binary segmentation in the case of
            `out_channels==1`. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        downsample_label_ratio (int): The ratio to downsample seg_label
            in losses. downsample_label_ratio > 1 will reduce memory usage.
            Disabled if downsample_label_ratio = 0.
            Default: 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 out_channels=None,
                 threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 downsample_label_ratio=0,
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
                 num_particles=50,
                 init_around_w_map=True,
                 logit2evidence='elu',
                 pow_alpha=True,
                 prior_variance=1.,
                 inner_loop_iters=100):
        super(BllSvgdBaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.use_bags = False
        self.frozen_features = False
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.downsample_label_ratio = downsample_label_ratio
        if not isinstance(self.downsample_label_ratio, int) or \
           self.downsample_label_ratio < 0:
            warnings.warn('downsample_label_ratio should '
                          'be set as an integer equal or larger than 0.')

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError('out_channels should be equal to num_classes,'
                             'except binary segmentation set out_channels == 1 and'
                             f'num_classes == 2, but got out_channels={out_channels}'
                             f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError('out_channels should be equal to num_classes,'
                             'except binary segmentation set out_channels == 1 and'
                             f'num_classes == 2, but got out_channels={out_channels}'
                             f'and num_classes={num_classes}')

        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        # Parameters for building SVGD
        self.num_particles = num_particles
        self.init_particles_around_w_map = init_around_w_map
        if logit2evidence.startswith('elu'):
            self.logit2evidence = elu_evidence
        elif logit2evidence.startswith('exp'):
            self.logit2evidence = exp_evidence
        else:
            raise NotImplementedError
        self.pow_alpha = pow_alpha
        self.prior_variance = prior_variance
        self.inner_loop_iters = inner_loop_iters

    def build_svgd(self):
        self.particles = torch.nn.Parameter(data=torch.randn(self.num_particles, self.ll_param_numel) * 1e-4)
        if self.init_particles_around_w_map is not None:
            w_map = torch.cat([self.conv_seg.weight.reshape(-1), self.conv_seg.bias.reshape(-1)]).detach()
            self.particles.data = w_map.reshape(1, -1).data + self.particles.data
        optim_ = torch.optim.SGD([self.particles], lr=1e-2)
        prior_ = tdist.Normal(loc=torch.zeros(self.ll_param_numel),
                              scale=torch.ones(self.ll_param_numel) * self.prior_variance)
        kernel_ = RBF()
        self.svgd = SVGD(prior_, kernel_, optim_)

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(input=x, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg_forward(feat)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        if self.downsample_label_ratio > 0:
            seg_label = seg_label.float()
            target_size = (seg_label.shape[2] // self.downsample_label_ratio,
                           seg_label.shape[3] // self.downsample_label_ratio)
            seg_label = resize(input=seg_label, size=target_size, mode='nearest')
            seg_label = seg_label.long()
        seg_logit = resize(input=seg_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        seg_label = seg_label.squeeze().repeat(self.num_particles, 1, 1)

        alpha = self.logit2evidence(seg_logit) + 1
        if self.pow_alpha:
            alpha = alpha**2
        loss['acc_seg'] = accuracy(alpha, seg_label, ignore_index=self.ignore_index)
        loss['edl_loss'] = -self.svgd.step(self.particles, alpha, seg_label)
        return loss


class SVGD:

    def __init__(self, prior, kernel, optimizer):
        self.kernel = kernel
        self.optim = optimizer
        self.prior = prior

    def tdist_to_device(self, device):
        mean = self.prior.loc.to(device)
        scale = self.prior.scale.to(device)
        self.prior = tdist.Normal(mean, scale)

    def phi(self, particles, alpha, target):

        particles = particles.detach().requires_grad_(True)
        # likelihood => P(data|theta) for EDL, could be changed to Multinomial
        target = target.unsqueeze(1)
        mask_ignore = (target == 255)
        target[mask_ignore] = 0
        one_hot_target = torch.zeros_like(alpha, dtype=torch.uint8).scatter_(1, target, 1)

        # Likelihood with EDL
        strength = torch.sum(alpha, dim=1, keepdim=True)
        # Eq. 3 at https://arxiv.org/abs/1806.01768 with fliped sign
        log_likelihood = -torch.sum(one_hot_target * (torch.log(strength) - torch.log(alpha)), axis=1, keepdims=True)

        log_likelihood = log_likelihood[~mask_ignore].mean()

        # Prior => Q(theta)
        logprior = self.prior.log_prob(particles).mean()

        log_prob = log_likelihood + logprior

        score_func = autograd.grad(log_prob.sum(), particles)[0]

        K_XX = self.kernel(particles, particles.detach())
        grad_K = -autograd.grad(K_XX.sum(), particles)[0]

        phi = (K_XX.detach().matmul(score_func) + grad_K) / particles.size(0)

        return phi, log_likelihood

    def step(self, particles, alpha, target):
        self.optim.zero_grad()
        update = self.phi(particles, alpha, target)
        particles.grad = -update[0]
        self.optim.step()
        return update[1]


class RBF(torch.nn.Module):

    def __init__(self, h=None):
        super(RBF, self).__init__()
        self.h = h

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.h is None:
            h = self.median(dnorm2.detach()) / (2 * torch.tensor(math.log(X.size(0))))
        else:
            h = self.h**2

        gamma = 1.0 / (2 * h)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


def elu_evidence(logits):
    return F.elu(logits) + 1


def exp_evidence(logits):
    return torch.exp(torch.clip(logits, -20., 20.))
