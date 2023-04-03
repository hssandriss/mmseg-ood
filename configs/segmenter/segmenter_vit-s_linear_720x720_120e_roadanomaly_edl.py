_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/cityscapes_720x720.py', '../_base_/epoch_runtime.py',
    '../_base_/schedules/schedule_120e.py'
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
        _delete_=True,
        type='FCNHead',
        in_channels=384,
        channels=384,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=19,
        loss_decode=dict(type='EDLLoss',
                         loss_variant='mse',
                         num_classes=19)))

optimizer = dict(lr=0.001, weight_decay=0.0)
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
)
