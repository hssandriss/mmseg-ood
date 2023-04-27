# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms import (BatchNorm, ConditionalHouseholder,
                                           ConditionalPlanar,
                                           ConditionalRadial, Householder,
                                           Planar, Radial, Sylvester,
                                           conditional_neural_autoregressive,
                                           neural_autoregressive, polynomial)
from pyro.nn import DenseNN

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BllBaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for NFBLLBaseDecodeHead.
    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
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
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg')),
                 density_type='flow',
                 kl_weight=1.,
                 flow_type='planar_flow',
                 flow_length=2,
                 use_bn_flow=False,
                 vi_nsamples_train=10,
                 vi_nsamples_test=1,
                 vi_use_lower_dim=True,
                 vi_latent_dim=1024,
                 initialize_at_w_map=True):
        super(BllBaseDecodeHead, self).__init__(init_cfg)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.use_bags = False
        self.frozen_features = False
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.use_bn_flow = use_bn_flow
        self.initialize_at_w_map = initialize_at_w_map
        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert seg_logist into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
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

        self.conv_seg = nn.Conv2d(
            channels, self.out_channels, kernel_size=1).train(False)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

        # Parameters for building NF
        self.vi_use_lower_dim = vi_use_lower_dim
        self.vi_latent_dim = vi_latent_dim
        self.density_type = density_type
        self.flow_type = flow_type
        self.kl_weight = kl_weight
        self.flow_length = flow_length
        self.vi_nsamples_train = vi_nsamples_train
        self.vi_nsamples_test = vi_nsamples_test

        # Tracking KL of VI
        self.kl_vals = []
        self.kl_weights = []
        self.epoch_num = 0
        self.total_epochs = 70

    def build_density_estimator(self):
        if not self.vi_use_lower_dim:
            self.latent_dim = self.ll_param_numel
        else:
            assert isinstance(
                self.vi_latent_dim, int
            ), "When using lower dim in density, 'vi_latent_dim' is required !"
            self.latent_dim = self.vi_latent_dim

        if self.density_type == 'flow':
            if self.flow_type in ('householder_flow', 'planar_flow',
                                  'radial_flow', 'sylvester_flow', 'naf_flow',
                                  'bnaf_flow', 'polynomial_flow'):
                self.density_estimation = DensityEstimation(
                    dim=self.latent_dim,
                    density_type=self.density_type,
                    flow_length=self.flow_length,
                    flow_type=self.flow_type,
                    use_bn=self.use_bn_flow)
            else:
                raise NotImplementedError
        elif self.density_type == 'conditional_flow':
            if self.flow_type in ('conditional_radial_flow',
                                  'conditional_planar_flow',
                                  'conditional_householder_flow',
                                  'conditional_naf_flow'):
                self.density_estimation = DensityEstimation(
                    dim=self.latent_dim,
                    density_type=self.density_type,
                    flow_length=self.flow_length,
                    flow_type=self.flow_type,
                    use_bn=self.use_bn_flow)
            else:
                raise NotImplementedError
        elif self.density_type in ('full_normal', 'fact_normal'):
            self.density_estimation = DensityEstimation(
                self.latent_dim, self.density_type)
        else:
            raise NotImplementedError

    def update_z0_params(self):
        """To run after loading weights It initializes the base distribution as
        a gaussian around previous MAP solution."""
        initial_z = torch.cat(
            [self.conv_seg.weight.reshape(-1),
             self.conv_seg.bias.reshape(-1)]).detach()
        if self.density_estimation.density_type in ('flow', 'cflow'):
            self.density_estimation.z0_mean = nn.Parameter(
                initial_z.data, requires_grad=False)
            self.density_estimation.base_dist = tdist.MultivariateNormal(
                self.density_estimation.z0_mean,
                self.density_estimation.z0_lvar.exp().diag())

        elif self.density_estimation.density_type in ('full_normal',
                                                      'fact_normal'):
            self.density_estimation.mu.data = initial_z.data
        else:
            raise NotImplementedError

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
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
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
        seg_logits, kl = self.forward(inputs, self.vi_nsamples_train)
        losses = self.losses(seg_logits, gt_semantic_seg, kl)
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
        seg_logits, _ = self.forward(inputs, self.vi_nsamples_test)
        return seg_logits

    def cls_seg(self, feat, z):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg_forward(feat, z)
        return output

    def cls_seg_x(self, feat, z):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg_forward_x(feat, z)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, kl):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        if self.density_estimation.density_type == 'conditional_flow':
            seg_label = seg_label.squeeze(1)
        else:
            seg_label = seg_label.squeeze(1).repeat(self.vi_nsamples_train, 1,
                                                    1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                # NLL
                if self.kl_weight == 'step':
                    kl_weight = float(
                        min(1., self.epoch_num / self.total_epochs))
                else:
                    assert isinstance(self.kl_weight,
                                      float), 'Invalid KL weights'
                    kl_weight = self.kl_weight
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit, seg_label, ignore_index=self.ignore_index)
                loss['loss_kld'] = kl_weight * kl
                self.kl_weights.append(kl_weight)
                self.kl_vals.append(kl.item())
                if loss_decode.loss_name.startswith('loss_edl'):
                    # load
                    logs = loss_decode.get_logs(seg_logit, seg_label,
                                                self.ignore_index)
                    loss.update(logs)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                if loss_decode.loss_name.startswith('loss_edl'):
                    raise NotImplementedError
        return loss


class DensityEstimation(nn.Module):

    def __init__(self, dim, density_type='flow', **kwargs):
        super(DensityEstimation, self).__init__()
        if density_type not in ('flow', 'conditional_flow', 'full_normal',
                                'fact_normal'):
            raise NotImplementedError
        self.dim = dim
        self.density_type = density_type
        if density_type == 'flow':
            self.flow_length = kwargs.pop('flow_length', 2)
            self.multi_scale_masks = [
                torch.ones(self.dim, dtype=torch.bool)
                for _ in range(self.flow_length)
            ]
            self.flow_type = kwargs.pop('flow_type', 'planar_flow')
            self.use_bn = kwargs.pop('use_bn', False)
            # Base gaussian distribution
            self.z0_mean = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            # build the flow sequence
            if self.flow_type == 'radial_flow':
                transforms = [
                    Radial(input_dim=self.dim) for _ in range(self.flow_length)
                ]
            elif self.flow_type == 'planar_flow':
                transforms = [
                    Planar(input_dim=self.dim) for _ in range(self.flow_length)
                ]
            elif self.flow_type == 'sylvester_flow':
                transforms = [
                    Sylvester(input_dim=self.dim, count_transforms=8)
                    for _ in range(self.flow_length)
                ]
            elif self.flow_type == 'householder_flow':
                transforms = [
                    Householder(input_dim=self.dim)
                    for _ in range(self.flow_length)
                ]
            elif self.flow_type == 'naf_flow':
                # Custom params:
                # transforms = [
                #     neural_autoregressive(
                #     input_dim=self.dim,
                #     hidden_dims=[3 * self.dim + 1] * self.flow_length,
                #     width=24)
                #     for i in range(self.flow_length)
                #     ]

                # Defaults:
                transforms = [
                    neural_autoregressive(input_dim=self.dim)
                    for _ in range(self.flow_length)
                ]

                # Use a multiscale-architecture (RealNVP style):
                # r = 2
                # current_dims = self.dim
                # for step in range(self.flow_length):
                #     self.multi_scale_masks[step][current_dims:] = False
                #     current_dims //= r
                # transforms = [
                #     neural_autoregressive(input_dim=int(mask.sum()))
                #     for mask in self.multi_scale_masks
                #     ]

            elif self.flow_type == 'bnaf_flow':
                raise NotImplementedError

            elif self.flow_type == 'polynomial_flow':
                # transforms = [
                #     polynomial(input_dim=self.dim,
                #                hidden_dims=[
                #                    10 * self.dim
                #                    for _ in range(self.flow_length)
                #                ]
                #                )
                # ]
                transforms = [
                    polynomial(input_dim=self.dim)
                    for _ in range(self.flow_length)
                ]
            else:
                raise NotImplementedError

            if self.use_bn:
                bn_indices = []
                for i in range(self.flow_length):
                    if i < self.flow_length - 1:
                        bn_indices.append(len(bn_indices) + i + 1)
                for i in bn_indices:
                    transforms.insert(i, BatchNorm(input_dim=self.dim))
            self.flow = nn.ModuleList(transforms)
            self.base_density = tdist.MultivariateNormal(
                loc=self.z0_mean, covariance_matrix=self.z0_lvar.exp().diag())
            self.prior_density = tdist.MultivariateNormal(
                loc=torch.zeros(self.dim),
                covariance_matrix=torch.ones(self.dim).diag())
        elif density_type == 'conditional_flow':
            self.feat_dims = kwargs.pop('feat_dims', 512)
            self.flow_length = kwargs.pop('flow_length', 2)
            self.flow_type = kwargs.pop('flow_type', 'cradial_flow')
            self.use_bn = kwargs.pop('use_bn', False)
            # Base gaussian distribution
            self.z0_mean = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            # build the flow sequence
            if self.flow_type == 'conditional_radial_flow':
                transforms = [
                    ConditionalRadial(
                        DenseNN(self.feat_dims,
                                [self.feat_dims, self.feat_dims],
                                [self.dim, 1, 1]))
                    for _ in range(self.flow_length)
                ]
            elif self.flow_type == 'conditional_planar_flow':
                transforms = [
                    ConditionalPlanar(
                        DenseNN(self.feat_dims,
                                [self.feat_dims, self.feat_dims],
                                [self.dim, 1, 1]))
                    for _ in range(self.flow_length)
                ]
            elif self.flow_type == 'conditional_householder_flow':
                transforms = [
                    ConditionalHouseholder(
                        DenseNN(self.feat_dims,
                                [self.feat_dims, self.feat_dims],
                                [self.dim, 1, 1]))
                    for _ in range(self.flow_length)
                ]

            elif self.flow_type == 'conditional_naf_flow':
                transforms = [
                    conditional_neural_autoregressive(
                        input_dim=self.dim, context_dim=self.feat_dims)
                    for _ in range(self.flow_length)
                ]
            else:
                raise NotImplementedError
            if self.use_bn:
                bn_indices = []
                for i in range(self.flow_length):
                    if i < self.flow_length - 1:
                        bn_indices.append(len(bn_indices) + i + 1)
                for i in bn_indices:
                    transforms.insert(i, BatchNorm(input_dim=self.dim))
            self.flow = nn.ModuleList(transforms)
            self.base_density = tdist.MultivariateNormal(
                loc=self.z0_mean, covariance_matrix=self.z0_lvar.exp().diag())
            self.prior_density = tdist.MultivariateNormal(
                loc=torch.zeros(self.dim),
                covariance_matrix=torch.ones(self.dim).diag())
        elif self.density_type == 'full_normal':
            # Reparametrization distribution N(0, I)
            self.z0_mean = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            # Target distribution parameters
            self.mu = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
            cov_numel = int(self.dim * (self.dim + 1) / 2)
            self.L_diag_elements = nn.Parameter(
                torch.ones(self.dim), requires_grad=True)
            self.L_udiag_elements = nn.Parameter(
                torch.ones(cov_numel - self.dim) * 1e-5, requires_grad=True)
        elif self.density_type == 'fact_normal':
            # Reparametrization distribution N(0, I)
            self.z0_mean = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(
                torch.zeros(self.dim), requires_grad=False)
            # Target distribution parameters
            self.mu = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
            cov_numel = int(self.dim)
            self.L_diag_elements = nn.Parameter(
                torch.ones(int(cov_numel)), requires_grad=True)
        else:
            raise NotImplementedError

    def tdist_to_device(self):
        device = next(self.parameters()).device
        if hasattr(self, 'base_density'):
            base_mean = self.base_density.loc
            base_cov = self.base_density.covariance_matrix
            base_mean, base_cov = base_mean.to(device), base_cov.to(device)
            self.base_density = tdist.MultivariateNormal(
                loc=base_mean, covariance_matrix=base_cov)
        if hasattr(self, 'prior_density'):
            prior_mean = self.prior_density.loc
            prior_cov = self.prior_density.covariance_matrix
            prior_mean, prior_cov = prior_mean.to(device), prior_cov.to(device)
            self.prior_density = tdist.MultivariateNormal(
                loc=prior_mean, covariance_matrix=prior_cov)

    def forward_normal(self, z, L):
        """
        Computes using reparametrization trick new z samples
        inputs:
            z: samples from N(0, I): Fixed distribution
            L: Lower triangular matrix or diagonal
        """
        if self.density_type == 'full_normal':
            assert self.L_diag_elements.numel() + self.L_udiag_elements.numel(
            ) == int(self.dim * (self.dim + 1) / 2)
            self._check_positive_definite(L)
            z = self.mu + z @ L
        elif self.density_type == 'fact_normal':
            assert self.L_diag_elements.numel() == self.dim
            z = self.mu + z @ L
        else:
            raise NotImplementedError
        return z

    def forward_flow(self, x, feats=None):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            feats: feature tensor (B x M).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        device = next(self.parameters()).device
        [B, _] = list(x.size())
        log_det = torch.zeros(B).to(device)
        for i in range(len(self.flow)):
            if isinstance(self.flow[i], ConditionalTransformModule):
                curr_flow_step = self.flow[i].condition(feats)
                x_next = curr_flow_step(x)
                inc = curr_flow_step.log_abs_det_jacobian(x, x_next)
                x = x_next
            elif isinstance(self.flow[i], TransformModule):
                x_next = torch.cat(
                    (self.flow[i](x[:, self.multi_scale_masks[i]]),
                     x[:, ~self.multi_scale_masks[i]]),
                    dim=-1)
                assert torch.isfinite(x_next).all()
                inc = self.flow[i].log_abs_det_jacobian(x, x_next)
                assert torch.isfinite(inc).all()
                if isinstance(self.flow[i], BatchNorm):
                    inc = inc.sum(-1)
                x = x_next
            else:
                raise NotImplementedError
            log_det = log_det + inc.squeeze()
        return x, log_det

    def sample_base(self, n):
        with torch.no_grad():
            device = next(self.parameters()).device
            std = torch.exp(.5 * self.z0_lvar)
            eps = torch.randn(size=[n, self.dim], device=device)
            z = eps.mul(std).add_(self.z0_mean)
        return z

    def flow_kl_loss(self, z0, zk, ldjs):
        bs = z0.size(0)
        kl = self.base_density.log_prob(
            z0) - ldjs - self.prior_density.log_prob(zk)
        assert kl.numel() == bs, 'KL term has a shape problem before averaging'
        return kl.mean()

    def flow_kl_loss_analytical(self, log_det):
        """Computes KL loss in closed form.

        Args:
            mean: mean of the gaussian approximate posterior.
            log_var: log-variance of the gaussian approximate posterior.
            log_det: log-determinant of the Jacobian.
        Returns: sum of KL loss over the minibatch.
        """
        lvar = self.z0_lvar.unsqueeze(0)
        mean = self.z0_mean.unsqueeze(0)
        kl = -.5 * torch.sum(
            1. + lvar - mean.pow(2) - lvar.exp(), dim=1, keepdim=True)
        return (kl - log_det).mean()

    def normal_kl_loss(self, L):
        # computed in closed form
        # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians # noqa
        self._check_positive_definite(L)
        kl = 0.5 * (-self._ldet_normal_cov(L) - self.dim +
                    self._normal_cov(L).trace() + self.mu.t() @ self.mu)
        return kl

    def _normal_cov(self, L):
        self._check_positive_definite(L)
        return L @ L.t()

    def _inv_normal_cov(self, L):
        self._check_positive_definite(L)
        return torch.cholesky_inverse(L)

    def _ldet_normal_cov(self, L):
        self._check_positive_definite(L)
        return 2 * (L.diag().log().sum() + 1e-6)

    def _check_positive_definite(self, L):
        # https://math.stackexchange.com/questions/462682/why-does-the-cholesky-decomposition-requires-a-positive-definite-matrix # noqa
        if not (L.diag() > 0).all():
            import ipdb
            ipdb.set_trace()

    @property
    def _L(self):
        """Reconstructs Lower Triangular Matrix."""
        full = True if self.density_type.startswith('full') else False
        device = next(self.parameters()).device
        assert self.L_diag_elements.numel() == self.dim
        L = F.softplus(self.L_diag_elements).diag() + 1e-05 * torch.eye(
            self.dim).to(device)
        if full:
            assert self.L_udiag_elements.numel() == int(self.dim *
                                                        (self.dim - 1) / 2)
            udiag_idx = torch.tril_indices(self.dim, self.dim, -1).to(device)
            L[udiag_idx[0, :], udiag_idx[1, :]] = self.L_udiag_elements
        return L


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = 0.5 * (-(x - mean)**2) * (
        -log_var).exp() - log_var - 2 * math.log(math.sqrt(2 * math.pi))

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm
