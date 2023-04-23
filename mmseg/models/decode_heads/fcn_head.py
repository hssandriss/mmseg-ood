# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .bll_decode_head import BllBaseDecodeHead
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class FCNBllHead(BllBaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))

        return feats

    def forward(self, inputs, nsamples):
        """Forward function."""
        # torch.cuda.synchronize()
        # t0 = time.time()
        with torch.no_grad():
            output, low_feats = self._forward_feature(inputs)
        # print(output.shape)
        assert output.is_leaf, 'you are backpropagating on feature extractor!'
        # torch.cuda.synchronize()
        # t1 = time.time()
        if nsamples == 1 and self.density_type == 'flow':
            # z0 = self.density_estimation.z0_mean.data.unsqueeze(0)
            z0 = self.density_estimation.sample_base(1)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(
                z0, low_feats)
            output = self.cls_seg_x(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            kl = -sum_log_jacobians.mean()
            # kl = self.density_estimation.flow_kl_loss(z0,
            #                                           zk, sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(
            #     sum_log_jacobians)
            return output, kl
        elif nsamples > 1 and self.density_type == 'flow':
            z0 = self.density_estimation.sample_base(nsamples)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(
                z0, low_feats)
            output = self.cls_seg_x(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            kl = self.density_estimation.flow_kl_loss(z0, zk,
                                                      sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(
            #     sum_log_jacobians)
            return output, kl
        elif nsamples == 1 and self.density_type == 'conditional_flow':
            z0 = self.density_estimation.z0_mean.data.unsqueeze(0)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(
                z0, low_feats)
            output = self.cls_seg(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            # kl = - sum_log_jacobians.mean()
            kl = self.density_estimation.flow_kl_loss(z0, zk,
                                                      sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(
            #     sum_log_jacobians)
            return output, kl
        elif nsamples > 1 and self.density_type == 'conditional_flow':
            z0 = self.density_estimation.sample_base(nsamples)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(
                z0, low_feats)
            if self.training:
                output = self.cls_seg(output, zk)
            else:
                output = self.cls_seg_x(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            # kl = - sum_log_jacobians.mean()
            kl = self.density_estimation.flow_kl_loss(z0, zk,
                                                      sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(
            #     sum_log_jacobians)

            return output, kl
        elif nsamples == 1 and self.density_type in ('full_normal',
                                                     'fact_normal'):
            L = self.density_estimation._L
            zk = self.density_estimation.mu.data.unsqueeze(0)
            kl = self.density_estimation.normal_kl_loss(L)
            output = self.cls_seg_x(output, zk)
            return output, kl
        elif nsamples > 1 and self.density_type in ('full_normal',
                                                    'fact_normal'):
            L = self.density_estimation._L
            z0 = self.density_estimation.sample_base(nsamples)
            zk = self.density_estimation.forward_normal(z0, L)
            kl = self.density_estimation.normal_kl_loss(L)
            output = self.cls_seg_x(output, zk)
            return output, kl
        else:
            raise NotImplementedError

    def conv_seg_forward_x(self, x, z):
        if self.vi_use_lower_dim:
            z = self.density_estimation_to_params(z)
            assert z.size(-1) == self.ll_param_numel
        z_list = torch.split(z, 1, 0)
        output = []
        # if not self.dropout.training:
        #     self.dropout.train()
        for z_ in z_list:
            dropout_x = self.dropout(x)
            # dropout_x = x
            z_ = z_.squeeze()
            output.append(
                F.conv2d(
                    input=dropout_x,
                    weight=z_[:self.w_numel].reshape(self.w_shape),
                    bias=z_[-self.b_numel:].reshape(self.b_shape)))
        return torch.cat(output, dim=0)

    def conv_seg_forward(self, x, z):
        if self.vi_use_lower_dim:
            z = self.density_estimation_to_params(z)
            assert z.size(-1) == self.ll_param_numel
        assert x.size(0) == z.size(0)
        z_list = torch.split(z, 1, 0)
        x_list = torch.split(x, 1, 0)
        output = []
        if not self.dropout.training:
            self.dropout.train()
        for x_, z_ in zip(x_list, z_list):
            dropout_x = self.dropout(x_)
            z_ = z_.squeeze()
            output.append(
                F.conv2d(
                    input=dropout_x,
                    weight=z_[:self.w_numel].reshape(self.w_shape),
                    bias=z_[-self.b_numel:].reshape(self.b_shape)))
        assert len(output) == z.size(0)
        return torch.cat(output, dim=0)
