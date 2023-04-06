_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/street_hazards_769x769.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(align_corners=True, num_classes=12),
    # auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(513, 513)))
