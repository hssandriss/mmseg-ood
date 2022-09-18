# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy
import torch.nn.functional as F

from pyro.distributions.transforms.sylvester import Sylvester
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from pyro.distributions.transforms.batchnorm import BatchNorm
import torch.distributions as tdist
import numpy as np
import pyro
"test w/: python tools/train.py configs/deeplabv3/deeplabv3_r50-d8_720x720_70e_cityscapes_nf_bll.py --experiment-tag 'TEST'"


class NfBllBaseDecodeHead(BaseModule, metaclass=ABCMeta):
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
                 init_cfg=dict(type='Normal', std=0.01, override=dict(name='conv_seg')),
                 density_type='flow',
                 kl_weight=1.,
                 flow_type='planar_flow',
                 flow_length=2,
                 use_bn_flow=False,
                 vi_nsamples_train=10,
                 vi_nsamples_test=1,
                 initialize_at_w_map=True):
        super(NfBllBaseDecodeHead, self).__init__(init_cfg)
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

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False
        # Parameters for building NF
        self.latent_dim = self.conv_seg.weight.numel() + self.conv_seg.bias.numel()
        self.density_type = density_type
        self.flow_type = flow_type
        self.kl_weight = kl_weight
        self.flow_length = flow_length
        self.vi_nsamples_train = vi_nsamples_train
        self.vi_nsamples_test = vi_nsamples_test
        if self.density_type == 'flow':
            if self.flow_type in ('iaf_flow', 'planar_flow', 'radial_flow', 'sylvester_flow'):
                self.density_estimation = DensityEstimation(
                    dim=self.latent_dim,
                    density_type=self.density_type,
                    kl_weight=self.kl_weight,
                    flow_length=self.flow_length,
                    flow_type=self.flow_type,
                    use_bn=self.use_bn_flow)
            else:
                raise NotImplementedError
        elif self.density_type in ('full_normal', 'fact_normal'):
            self.density_estimation = DensityEstimation(
                self.latent_dim,
                self.density_type,
                kl_weight=self.kl_weight)
        else:
            raise NotImplementedError
        self.w_shape, self.b_shape = self.conv_seg.weight.shape, self.conv_seg.bias.shape
        self.w_numel, self.b_numel = self.conv_seg.weight.numel(), self.conv_seg.bias.numel()
        self.initial_p = torch.cat([self.conv_seg.weight.reshape(-1), self.conv_seg.bias.reshape(-1)]).detach()

    def update_z0_params(self):
        """
        To run after loading weights
        It initializes the base distribution as a gaussian aroud previous MAP solution
        """
        initial_z = torch.cat([self.conv_seg.weight.reshape(-1), self.conv_seg.bias.reshape(-1)]).detach()
        if self.density_estimation.density_type == 'flow':
            self.density_estimation.z0_mean = nn.Parameter(initial_z.data, requires_grad=False)
            # self.density_estimation.base_dist = tdist.MultivariateNormal(self.density_estimation.z0_mean, self.density_estimation.z0_cov)
        elif self.density_estimation.density_type in ('full_normal', 'fact_normal'):
            self.density_estimation.mu.data = initial_z.data
        else:
            raise NotImplementedError

    def conv_seg_forward(self, x, z):
        z_list = torch.split(z, 1, 0)
        output = []
        for z_ in z_list:
            z_ = z_.squeeze()
            output.append(F.conv2d(input=x, weight=z_[:self.w_numel].reshape(self.w_shape), bias=z_[-self.b_numel:].reshape(self.b_shape)))
        return torch.cat(output, dim=0)

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
            # Here I can add weight normalization
        output = self.conv_seg_forward(feat, z)
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
        seg_label = seg_label.squeeze(1).repeat(self.vi_nsamples_train, 1, 1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        loss['acc_seg'] = accuracy(seg_logit, seg_label, ignore_index=self.ignore_index)

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                # NLL
                loss[loss_decode.loss_name] = loss_decode(seg_logit, seg_label, ignore_index=self.ignore_index) + kl
                loss['kl_term'] = kl.detach().data
                if loss_decode.loss_name.startswith("loss_edl"):
                    # load
                    logs = loss_decode.get_logs(seg_logit,
                                                seg_label,
                                                self.ignore_index)
                    loss.update(logs)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                if loss_decode.loss_name.startswith("loss_edl"):
                    raise NotImplementedError

        return loss


class DensityEstimation(nn.Module):
    def __init__(self, dim, density_type='flow', kl_weight=1., **kwargs):
        super(DensityEstimation, self).__init__()
        if density_type not in ('flow', 'full_normal', 'fact_normal'):
            raise NotImplementedError
        self.dim = dim
        self.density_type = density_type
        self.kl_weight = kl_weight
        if density_type == 'flow':
            self.flow_length = kwargs.pop('flow_length', 2)
            self.flow_type = kwargs.pop('flow_type', 'planar_flow')
            self.use_bn = kwargs.pop('use_bn', False)

            # Base gaussian distribution
            self.z0_mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(torch.ones(self.dim), requires_grad=False)
            # self.base_dist = tdist.MultivariateNormal(self.z0_mean, self.z0_cov)

            # build the flow sequence
            if self.flow_type == 'radial_flow':
                transforms = [Radial(dim) for _ in range(self.flow_length)]
            elif self.flow_type == 'sylvester_flow':
                transforms = [Sylvester(dim) for _ in range(self.flow_length)]
            elif self.flow_type == 'planar_flow':
                transforms = [Planar(dim) for _ in range(self.flow_length)]
            elif self.flow_type == 'iaf_flow':
                transforms = [affine_autoregressive(dim, hidden_dims=[self.dim]) for _ in range(self.flow_length)]
            else:
                raise NotImplementedError
            if self.use_bn:
                bn_indices = []
                for i in range(self.flow_length):
                    if i < self.flow_length - 1:
                        bn_indices.append(len(bn_indices) + i + 1)
                for i in bn_indices:
                    transforms.insert(i, BatchNorm(self.dim))
            self.transforms = nn.Sequential(*transforms)
            # self.flow_dist = pyro.distributions.TransformedDistribution(self.base_dist, transforms)
        elif self.density_type == 'full_normal':
            # Reparametrization distribution
            self.z0_mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(torch.eye(self.dim), requires_grad=False)
            # self.base_dist = tdist.MultivariateNormal(self.z0_mean, self.z0_lvar)
            # Target distribution parameters
            self.mu = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
            cov_numel = int(self.dim * (self.dim + 1) / 2)
            self.L_diag_elements = nn.Parameter(torch.ones(self.dim), requires_grad=True) * 0.1
            self.L_udiag_elements = nn.Parameter(torch.ones(cov_numel - self.dim), requires_grad=True) * 0.1
            self.udiag_idx = torch.tril_indices(self.dim, self.dim, -1)
            self.diag_idx = torch.vstack((torch.arange(self.dim), torch.arange(self.dim)))
            # More efficient than reconsructing every time
            L = torch.zeros((self.dim, self.dim))
            L[self.udiag_idx[0, :], self.udiag_idx[1, :]] = self.L_udiag_elements
            L[self.diag_idx[0, :], self.diag_idx[1, :]] = F.softplus(self.L_diag_elements) + 0.01
            # ! Bug: getting covariance not positive definite ==> More debugging
            self.target_dist = tdist.MultivariateNormal(self.mu, L @ L.t())
        elif self.density_type == 'fact_normal':
            # Reparametrization distribution
            self.z0_mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
            self.z0_lvar = nn.Parameter(torch.eye(self.dim), requires_grad=False)
            # self.base_dist = tdist.MultivariateNormal(self.z0_mean, self.z0_cov)
            # Target distribution parameters
            cov_numel = int(self.dim)
            self.mu = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
            self.L_diag_elements = nn.Parameter(torch.ones(int(cov_numel)), requires_grad=True)
            # More efficient than reconsructing every time
            # self.target_dist = tdist.MultivariateNormal(self.mu, self.L_diag_elements.diag())
        else:
            raise NotImplementedError

    def forward_normal(self, z):
        """
        Computes using reparametrization trick new z samples
        inputs:
            z: samples from N(0, I): Fixed distribution
        """
        if self.density_type == 'full_normal':
            assert self.L_diag_elements.numel() + self.L_udiag_elements.numel() == int(self.dim * (self.dim + 1) / 2)
            z = self.mu + z @ self._L(full=True)
        elif self.density_type == 'fact_normal':
            assert self.L_diag_elements.numel() == self.dim
            z = self.mu + z @ self._L(full=False)
        return z, None

    def forward_flow(self, z):
        # TODO: Reimplement and use TransformedDistribution API (less errors)
        # np.savetxt('initial_5000_z_samples_base_I.csv', z.detach().cpu().numpy())
        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            if isinstance(transform, BatchNorm):
                sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z_next, z).sum(-1)
            else:
                sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z_next, z)
            z = z_next
        # np.savetxt("trained_5000_z_samples_2xradial.csv", z.detach().cpu().numpy())
        # proj = self.pca.transform(z.detach().cpu().numpy())
        # clusters = self.kmeans.predict(proj)
        return z, sum_log_jacobians

    def sample_base(self, n):
        device = next(self.parameters()).device
        std = torch.exp(.5 * self.z0_lvar)
        eps = torch.randn(size=[n, self.dim], device=device)
        z = eps.mul(std).add_(self.z0_mean)
        return z

    def log_prob_flow(self, z):
        z, sum_log_jacobians = self.forward_flow(z)
        log_prob_z = self.base_dist.log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians
        return log_prob_x

    def log_prob_normal(self, z):
        z, _ = self.forward_normal(z)
        assert all(self.target_dist.mean.data == self.mu.data), "target_dist parameters are not getting updated"
        return self.target_dist.log_prob(z)

    def flow_kl_loss(self, z0, zk, ldjs):
        # ln p(z_k)  (not averaged)
        log_p_zk = log_normal_standard(zk, dim=1)
        # ln q(z_0)  (not averaged)
        log_q_z0 = log_normal_diag(z0, mean=self.z0_mean, log_var=self.z0_lvar, dim=1)

        # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        logs = log_q_z0 - log_p_zk
        kl = logs - ldjs
        return self.kl_weight * kl.mean()

    def normal_kl_loss(self, full=True):
        # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        kl = 0.5 (-self._ldet_normal_cov(full=full) - self.dim + self._normal_cov(full=full).trace() + self.mu.t()  @ self.mu)
        return kl

    def _normal_cov(self, full=True):
        assert self._check_positive_definite()
        return self._L(full=full) @ self._L(full=full).t()

    def _inv_normal_cov(self, full=True):
        assert self._check_positive_definite()
        return torch.cholesky_inverse(self._L(full=full))

    def _ldet_normal_cov(self, full=True):
        return self._L(full=full).diag().prod().log()

    def _check_positive_definite(self, full=True):
        assert (self._L(full=full).diag() > 0).all(), "The matrix L is not positive definite"

    def _L(self, full=True):
        assert self.L_diag_elements.numel() + self.L_udiag_elements.numel() == int(self.dim * (self.dim + 1) / 2)
        L = torch.zeros((self.dim, self.dim))
        L[self.diag_idx] = torch.exp(self.L_diag_elements) + 1e-05
        if full:
            L[self.udiag_idx] = self.L_udiag_elements
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
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm
