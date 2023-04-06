_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/street_hazards_512x512_val.py',
    '../_base_/epoch_runtime.py', '../_base_/schedules/schedule_120e.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'  # noqa
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(720, 720),
        embed_dims=384,
        num_heads=6,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=384,
        channels=384,
        num_classes=12,
        num_layers=2,
        num_heads=6,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(type='EDLLoss', num_classes=12)))

optimizer = dict(lr=0.001, weight_decay=0.0)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)
