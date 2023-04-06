# dataset settings
dataset_type = 'StreetHazardsDataset'
data_root = '/misc/lmbraid17/datasets/public/StreetHazards/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
max_ratio = 2
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2049, 1025),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images/validation',
        ann_dir='train/annotations/validation',
        seg_map_suffix='.png',
        img_suffix='.png',
        pipeline=test_pipeline,
    ),
)
