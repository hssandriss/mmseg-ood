_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/cityscapes_720x720.py', '../_base_/epoch_runtime.py',
    '../_base_/schedules/schedule_70e_constant.py'
]
model = dict(
    decode_head=dict(
        type='ASPPBllHead',
        align_corners=True,
        num_classes=19,
        loss_decode=dict(type='EDLLoss', num_classes=19)), )
data = dict(samples_per_gpu=4, workers_per_gpu=4)
