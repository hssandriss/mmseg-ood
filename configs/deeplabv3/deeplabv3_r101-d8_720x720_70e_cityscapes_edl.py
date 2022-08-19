_base_ = './deeplabv3_r50-d8_720x720_70e_cityscapes_edl.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
data = dict(samples_per_gpu=3,
            workers_per_gpu=3)
