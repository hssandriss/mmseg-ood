# yapf:disable
log_config = dict(
    interval=1,
    by_epoch=True,
    hooks=[
        dict(type='CustomTextLoggerHook'),
        dict(type='CustomTensorboardLoggerHook')
    ]
)
custom_hooks = [
    dict(type='ParseEpochToDecodeHead', priority='NORMAL'),
    dict(type='ParseEpochToLossHook', priority='NORMAL'),
    # dict(type='DetectAnomalyHook', priority='NORMAL'),
    # dict(type='CustomEMAHook', warm_up_epochs=10, priority='NORMAL'),
]
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
