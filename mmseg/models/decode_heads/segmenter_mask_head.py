# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import ModuleList

from mmseg.models.backbones.vit import TransformerEncoderLayer
from ..builder import HEADS
from .bll_decode_head import BllBaseDecodeHead
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SegmenterMaskTransformerHead(BaseDecodeHead):
    """Segmenter: Transformer for Semantic Segmentation.

    This head is the implementation of
    `Segmenter:　<https://arxiv.org/abs/2105.05633>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input image.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        embed_dims (int): The number of embedding dimension.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            in_channels,
            num_layers,
            num_heads,
            embed_dims,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        super(SegmenterMaskTransformerHead, self).__init__(
            in_channels=in_channels, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.dec_proj = nn.Linear(in_channels, embed_dims)

        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=2)[1]

        self.init_std = init_std

        delattr(self, 'conv_seg')

    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        trunc_normal_init(self.patch_proj, std=self.init_std)
        trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        x = self.dec_proj(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for layer in self.layers:
            x = layer(x)
        x = self.decoder_norm(x)

        patches = self.patch_proj(x[:, :-self.num_classes])
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        return masks


@HEADS.register_module()
class SegmenterMaskTransformerBllHead(BllBaseDecodeHead):
    """Segmenter: Transformer for Semantic Segmentation.

    This head is the implementation of
    `Segmenter:　<https://arxiv.org/abs/2105.05633>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input image.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        embed_dims (int): The number of embedding dimension.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            in_channels,
            num_layers,
            num_heads,
            embed_dims,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        super(SegmenterMaskTransformerBllHead, self).__init__(
            in_channels=in_channels, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.dec_proj = nn.Linear(in_channels, embed_dims)

        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=2)[1]

        self.init_std = init_std

        self.patch_proj_w_shape = self.patch_proj.weight.shape
        self.classes_proj_shape = self.classes_proj.weight.shape
        self.patch_proj_numel = self.patch_proj.weight.numel()
        # self.classes_proj_numel = self.classes_proj.weight.numel()

        # self.ll_param_numel = self.patch_proj_numel + self.classes_proj_numel
        self.ll_param_numel = self.patch_proj_numel

        self.density_estimation_to_params = nn.Linear(
            self.vi_latent_dim, self.ll_param_numel, bias=False)
        self.build_density_estimator()
        delattr(self, 'conv_seg')
        delattr(self, 'patch_proj')
        # delattr(self, 'classes_proj')

    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        # trunc_normal_init(self.patch_proj, std=self.init_std)
        # trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs, nsamples):
        with torch.no_grad():
            x = self._transform_inputs(inputs)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

            x = self.dec_proj(x)
            cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
            x = torch.cat((x, cls_emb), 1)
            for layer in self.layers:
                x = layer(x)
            x = self.decoder_norm(x)
        input_dims = (b, c, h, w)
        if nsamples == 1 and self.density_type == 'flow':
            z0 = self.density_estimation.sample_base(1)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(
                z0, None)
            kl = -sum_log_jacobians.mean()
            output = self.seg_forward(x, zk, input_dims)
            return output, kl
        elif nsamples > 1 and self.density_type == 'flow':
            z0 = self.density_estimation.sample_base(nsamples)
            zk, sum_log_jacobians = self.density_estimation.forward_flow(z0)
            kl = self.density_estimation.flow_kl_loss(z0, zk,
                                                      sum_log_jacobians)
            output = self.seg_forward_x(x, zk, input_dims)
            return output, kl
        elif nsamples == 1 and self.density_type in ('full_normal',
                                                     'fact_normal'):
            L = self.density_estimation._L
            zk = self.density_estimation.mu.data.unsqueeze(0)
            kl = self.density_estimation.normal_kl_loss(L)
            output = self.seg_forward(x, zk, input_dims)
            return output, kl
        elif nsamples > 1 and self.density_type in ('full_normal',
                                                    'fact_normal'):
            L = self.density_estimation._L
            z0 = self.density_estimation.sample_base(nsamples)
            zk = self.density_estimation.forward_normal(z0, L)
            kl = self.density_estimation.normal_kl_loss(L)
            output = self.seg_forward_x(x, zk, input_dims)
            return output, kl
        elif nsamples == 1 and self.density_type == 'conditional_flow':
            raise NotImplementedError
        elif nsamples > 1 and self.density_type == 'conditional_flow':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def seg_forward_x(self, feats, z, input_dims):
        # Results into bs = feats.size(0)*z.size(0)
        b, c, h, w = input_dims
        # Force activate dropout during test
        # if not self.dropout.training:
        #     self.dropout.train()
        if self.vi_use_lower_dim:
            z = self.density_estimation_to_params(z)
            assert z.size(-1) == self.ll_param_numel
        z_list = torch.split(z, 1, 0)
        output = []
        for z_ in z_list:
            if self.dropout and self.dropout.training:
                feats = self.dropout(feats)
            self.patch_proj_w = z_.squeeze().reshape(self.patch_proj_w_shape)
            # self.classes_proj_w  = z_[self.patch_proj_numel:].reshape(
            #     self.classes_proj_shape)
            patches = F.linear(feats[:, :-self.num_classes], self.patch_proj_w,
                               None)

            # cls_seg_feat = F.linear(feats[:, -self.num_classes:],
            #                         self.classes_proj_w, None)
            cls_seg_feat = self.classes_proj(feats[:, -self.num_classes:])

            patches = F.normalize(patches, dim=2, p=2)
            cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)
            masks = patches @ cls_seg_feat.transpose(1, 2)
            masks = self.mask_norm(masks)
            masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)
            output.append(masks)
        return torch.cat(output, dim=0)

    def seg_forward(self, feats, z, input_dims):
        # Results into bs = feats.size(0)
        b, c, h, w = input_dims
        # Force activate dropout during test
        # if not self.dropout.training:
        #     self.dropout.train()
        if self.vi_use_lower_dim:
            z = self.density_estimation_to_params(z)
            assert z.size(-1) == self.ll_param_numel
        assert feats.size(0) == z.size(0)
        z_list = torch.split(z, 1, 0)
        feats_list = torch.split(feats, 1, 0)
        output = []
        for x_, z_ in zip(feats_list, z_list):
            if self.dropout and self.dropout.training:
                x_ = self.dropout(x_)
            z_ = z_.squeeze()
            self.patch_proj_w = z_[:self.patch_proj_numel].reshape(
                self.patch_proj_w_shape)
            # self.classes_proj_w  = x_[self.patch_proj_numel:].reshape(
            #     self.classes_proj_shape)
            patches = F.linear(x_[:, :-self.num_classes], self.patch_proj_w,
                               None)
            # cls_seg_feat = F.linear(x_[:, -self.num_classes:],
            #                         self.classes_proj_w, None)
            cls_seg_feat = self.classes_proj(x_[:, -self.num_classes:])

            patches = F.normalize(patches, dim=2, p=2)
            cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)
            masks = patches @ cls_seg_feat.transpose(1, 2)
            masks = self.mask_norm(masks)
            masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)
            output.append(masks)
        assert len(output) == z.size(0)
        return torch.cat(output, dim=0)
