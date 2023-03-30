# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .bll_decode_head import BllBaseDecodeHead
import torch.nn.functional as F


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
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
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        if self.frozen_features:
            assert output.is_leaf
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class ASPPBllHead(BllBaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPBllHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.w_shape, self.b_shape = self.conv_seg.weight.shape, self.conv_seg.bias.shape
        self.w_numel, self.b_numel = self.conv_seg.weight.numel(), self.conv_seg.bias.numel()
        self.ll_param_numel = self.w_numel + self.b_numel
        self.density_estimation_to_params = nn.Linear(self.vi_latent_dim, self.ll_param_numel, bias=False)
        self.build_density_estimator()

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
        # import ipdb; ipdb.set_trace()
        low_feats = self.image_pool(x)
        aspp_outs = [
            resize(
                low_feats,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        return feats, low_feats.squeeze()

    def forward(self, inputs, nsamples):
        """Forward function."""
        # torch.cuda.synchronize()
        # t0 = time.time()
        with torch.no_grad():
            output, low_feats = self._forward_feature(inputs)
        # print(output.shape)
        assert output.is_leaf, "you are backpropagating on feature extractor !"
        # torch.cuda.synchronize()
        # t1 = time.time()
        if nsamples == 1 and self.density_type == 'flow':
            # z0 = self.density_estimation.z0_mean.data.unsqueeze(0)
            z0 = self.density_estimation.sample_base(1)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(z0, low_feats)
            output = self.cls_seg_x(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            kl = - sum_log_jacobians.mean()
            # kl = self.density_estimation.flow_kl_loss(z0, zk, sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(sum_log_jacobians)
            return output, kl
        elif nsamples > 1 and self.density_type == 'flow':
            z0 = self.density_estimation.sample_base(nsamples)
            # joblib.dump(z0.detach().cpu().numpy(), 'base_samples.pkl')
            # import ipdb; ipdb.set_trace()
            # z0 = torch.as_tensor(joblib.load('base_samples.pkl')).to(output.device)

            zk, sum_log_jacobians = self.density_estimation.forward_flow(z0, low_feats)
            # joblib.dump(zk.detach().cpu().numpy(), 'naf_samples.pkl')
            # print("Variance of z >>>>>", zk.var(0).mean())

            output = self.cls_seg_x(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            # kl = - sum_log_jacobians.mean()
            kl = self.density_estimation.flow_kl_loss(z0, zk, sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(sum_log_jacobians)
            return output, kl
        elif nsamples == 1 and self.density_type == 'conditional_flow':
            z0 = self.density_estimation.z0_mean.data.unsqueeze(0)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(z0, low_feats)
            output = self.cls_seg(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            # kl = - sum_log_jacobians.mean()
            kl = self.density_estimation.flow_kl_loss(z0, zk, sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(sum_log_jacobians)
            return output, kl
        elif nsamples > 1 and self.density_type == 'conditional_flow':
            z0 = self.density_estimation.sample_base(nsamples)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(z0, low_feats)
            if self.training:
                output = self.cls_seg(output, zk)
            else:
                output = self.cls_seg_x(output, zk)
            # Reverse KLD: https://arxiv.org/abs/1912.02762 page 7 Eq. 17-18
            # kl = - sum_log_jacobians.mean()
            kl = self.density_estimation.flow_kl_loss(z0, zk, sum_log_jacobians)
            # kl = self.density_estimation.flow_kl_loss_analytical(sum_log_jacobians)

            return output, kl
        elif nsamples == 1 and self.density_type in ('full_normal', 'fact_normal'):
            L = self.density_estimation._L
            zk = self.density_estimation.mu.data.unsqueeze(0)
            kl = self.density_estimation.normal_kl_loss(L)
            output = self.cls_seg_x(output, zk)
            return output, kl
        elif nsamples > 1 and self.density_type in ('full_normal', 'fact_normal'):
            L = self.density_estimation._L
            z0 = self.density_estimation.sample_base(nsamples)
            # z0 = torch.as_tensor(joblib.load('base_samples.pkl')).to(output.device)
            zk = self.density_estimation.forward_normal(z0, L)
            # joblib.dump(zk.detach().cpu().numpy(), 'full_normal_samples.pkl')
            # print("Variance of z >>>>>", zk.var(0).mean())
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
            output.append(F.conv2d(input=dropout_x, weight=z_[:self.w_numel].reshape(self.w_shape), bias=z_[-self.b_numel:].reshape(self.b_shape)))
        # import ipdb; ipdb.set_trace()
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
            output.append(F.conv2d(input=dropout_x, weight=z_[:self.w_numel].reshape(self.w_shape), bias=z_[-self.b_numel:].reshape(self.b_shape)))
        assert len(output) == z.size(0)
        return torch.cat(output, dim=0)
