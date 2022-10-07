# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook, build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import (build_ddp, build_dp, find_latest_checkpoint,
                         get_root_logger)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on devices
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # DDP wrapper
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner

    # Check if we are applying BLL
    is_bll = 'bll' in cfg.model.decode_head.type.lower()

    if is_bll:
        optim_type = cfg.optimizer.pop('type')
        assert optim_type == 'SGD', "currently just SGD is supported for BLL models"
        optimizer = torch.optim.SGD(params=[p for name, p in model.named_parameters() if "density_estimation" in name], **cfg.optimizer)
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    # Freeze up to last layer
    freeze_features = meta.pop("freeze_features", False)
    # Freeze up the encoder
    freeze_encoder = meta.pop("freeze_encoder", False)
    # reinitialize the last layer parameters
    init_not_frozen = meta.pop("init_not_frozen", False)

    assert not (freeze_encoder and freeze_features), "freeze encoder and freeze features are mutually exclusive"
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    if cfg.device == 'npu':
        optimiter_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
        cfg.optimizer_config = optimiter_config if \
            not cfg.optimizer_config else cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        val_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('val_dataloader', {}),
        }
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    if is_bll:
        assert cfg.load_from or cfg.resume_from, "It is required to pre-learned features for BLL"

    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from

    if cfg.resume_from and is_bll:
        runner.resume(cfg.resume_from)
        runner.model.module.freeze_encoder()
        runner.model.module.freeze_feature_extractor()
    elif cfg.resume_from and not is_bll:
        runner.resume(cfg.resume_from)
    elif cfg.load_from and is_bll:
        runner.load_checkpoint(cfg.load_from)
        runner.model.module.freeze_encoder()
        runner.model.module.freeze_decoder_except_density_estimation()
    elif cfg.load_from and not is_bll:
        runner.load_checkpoint(cfg.load_from)
        if freeze_encoder and init_not_frozen:
            ckpt = torch.load(cfg.load_from)
            to_keep = [k for k in ckpt["state_dict"].keys() if k.startswith('backbone')]
            to_delete = [k for k in ckpt["state_dict"].keys() if not k.startswith('backbone')]
            for k in to_delete:
                del ckpt["state_dict"][k]
            cfg.load_from = os.path.join(cfg.work_dir, "src.pth")
            torch.save(ckpt, cfg.load_from)
        if freeze_features and init_not_frozen:
            ckpt = torch.load(cfg.load_from)
            to_delete = [k for k in ckpt["state_dict"].keys() if k.startswith("decode_head.conv_seg")]
            to_keep = [k for k in ckpt["state_dict"].keys() if not k.startswith('decode_head.conv_seg')]
            for k in to_delete:
                del ckpt["state_dict"][k]
            cfg.load_from = os.path.join(cfg.work_dir, "src.pth")
            torch.save(ckpt, cfg.load_from)
        runner.load_checkpoint(cfg.load_from)
    else:
        pass

    if is_bll and cfg.model.decode_head.density_type in ("flow", "cflow"):
        if runner.model.module.decode_head.initialize_at_w_map:
            runner.model.module.decode_head.update_z0_params()
        runner.model.module.decode_head.density_estimation.tdist_to_device()

    runner.run(data_loaders, cfg.workflow)
