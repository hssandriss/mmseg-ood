# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
import matplotlib.pyplot as plt
from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
from torch.utils.tensorboard import SummaryWriter
import shutil
import json
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')

    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument("--use-bags", action='store_true', help='determines weather to use bags of predictors')
    parser.add_argument("--bags-mul", type=int, default=10, help='determines weather to use bags of predictors')
    parser.add_argument("--all", action='store_true', help='determines weather to test all checkpoints')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    # import warnings; warnings.filterwarnings("ignore")
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    if args.use_bags:
        dataset.get_bags(mul=args.bags_mul)
        print(dataset.bag_class_counts)
        cfg.model.decode_head.num_classes += dataset.num_bags
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    all_checkpoints = [os.path.join(args.work_dir, file) for file in os.listdir(args.work_dir) if file.endswith(".pth") and file != "latest.pth"]
    all_checkpoints.sort(key=lambda file: int(re.search(r"(?:epoch)_([0-9]+).(?:pth)$", osp.basename(file)).groups()[0]))
    all_checkpoints_iter = [int(re.search(r"(?:epoch)_([0-9]+).(?:pth)$", osp.basename(file)).groups()[0]) for file in all_checkpoints]
    reg_ood_summary = pd.DataFrame(
        columns=["epoch", 'max_prob.auroc', 'max_prob.aupr', 'max_prob.fpr95', 'max_logit.auroc', 'max_logit.aupr',
                 'max_logit.fpr95', 'emp_entropy.auroc', 'emp_entropy.aupr', 'emp_entropy.fpr95'])
    sl_ood_summary = pd.DataFrame(columns=["epoch", 'u.auroc', 'u.aupr', 'u.fpr95', 'disonnance.auroc',
                                           'disonnance.aupr', 'disonnance.fpr95', 'dir_entropy.auroc', 'dir_entropy.aupr', 'dir_entropy.fpr95'])

    if not args.all:
        index_last = max(range(len(all_checkpoints_iter)), key=all_checkpoints_iter.__getitem__)
        all_checkpoints = [all_checkpoints[index_last]]
        all_checkpoints_iter = [all_checkpoints_iter[index_last]]
    if args.all:
        # clean previous test logs
        if osp.exists(os.path.join(args.work_dir, "tf_logs_test")):
            shutil.rmtree(os.path.join(args.work_dir, "tf_logs_test"))
        writer = SummaryWriter(log_dir=os.path.join(args.work_dir, "tf_logs_test"))
    ans = []

    for i, checkpoint in enumerate(all_checkpoints):
        # build the model and load checkpoint
        if args.show_dir:
            args.show_dir = args.show_dir + f"_{all_checkpoints_iter[i]}"
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        # import torch.nn as nn
        # model.decode_head.conv_seg = nn.utils.weight_norm(
        #     nn.Conv2d(model.decode_head.channels, model.decode_head.num_classes, kernel_size=1), "weight", dim=1)

        # with torch.no_grad():
        #     model.decode_head.conv_seg.weight.div_(torch.norm(model.decode_head.conv_seg.weight, dim=1, keepdim=True))

        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            print('"PALETTE" not found in meta, use dataset.PALETTE instead')
            model.PALETTE = dataset.PALETTE
        if args.use_bags:
            setattr(model.decode_head, "use_bags", True)
            setattr(model.decode_head, "bags_kwargs", dict(
                num_bags=dataset.num_bags,
                label2bag=dataset.label2bag,
                bag_label_maps=dataset.bag_label_maps,
                bag_masks=dataset.bag_masks,
                bags_classes=dataset.bags_classes,
                bag_class_counts=dataset.bag_class_counts
            ))
        else:
            setattr(model.decode_head, "use_bags", False)

        # clean gpu memory when starting a new evaluation.
        torch.cuda.empty_cache()
        eval_kwargs = {} if args.eval_options is None else args.eval_options

        # Deprecated
        efficient_test = eval_kwargs.get('efficient_test', False)
        if efficient_test:
            warnings.warn(
                '``efficient_test=True`` does not have effect in tools/test.py, '
                'the evaluation and format results are CPU memory efficient by '
                'default')

        eval_on_format_results = (
            args.eval is not None and 'cityscapes' in args.eval)
        if eval_on_format_results:
            assert len(args.eval) == 1, 'eval on format results is not ' \
                                        'applicable for metrics other than ' \
                                        'cityscapes'
        if args.format_only or eval_on_format_results:
            if 'imgfile_prefix' in eval_kwargs:
                tmpdir = eval_kwargs['imgfile_prefix']
            else:
                tmpdir = '.format_cityscapes'
                eval_kwargs.setdefault('imgfile_prefix', tmpdir)
            mmcv.mkdir_or_exist(tmpdir)
        else:
            tmpdir = None

        cfg.device = get_device()
        if not distributed:
            warnings.warn(
                'SyncBN is only supported with DDP. To be compatible with DP, '
                'we convert SyncBN to BN. Please use dist_train.sh which can '
                'avoid this error.')
            if not torch.cuda.is_available():
                assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                    'Please use MMCV >= 1.4.4 for CPU training!'
            model = revert_sync_batchnorm(model)
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            results = single_gpu_test(
                model,
                data_loader,
                args.show,
                args.show_dir,
                False,
                args.opacity,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)
            results = multi_gpu_test(
                model,
                data_loader,
                args.tmpdir,
                args.gpu_collect,
                False,
                pre_eval=args.eval is not None and not eval_on_format_results,
                format_only=args.format_only or eval_on_format_results,
                format_args=eval_kwargs)

        # prev_map = torch.cat([model.module.decode_head.conv_seg.weight.reshape(-1), model.module.decode_head.conv_seg.bias.reshape(-1)]).detach().data
        # curr_map = model.module.decode_head.density_estimation.z0_mean.data
        # assert torch.eq(prev_map, curr_map).all(), 'the base dist mean does not coincide with previous MAP'
        # import ipdb; ipdb.set_trace()
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                warnings.warn(
                    'The behavior of ``args.out`` has been changed since MMSeg '
                    'v0.16, the pickled outputs could be seg map as type of '
                    'np.array, pre-eval results or file paths for '
                    '``dataset.format_results()``.')
                print(f'\nwriting results to {args.out}')
                mmcv.dump(results, args.out)
            if args.eval:
                eval_kwargs.update(metric=args.eval)
                metric = dataset.evaluate(results, **eval_kwargs)
                metric_dict = dict(config=args.config, metric=metric)
                metric_dict["iter"] = all_checkpoints_iter[i]
                mmcv.dump(metric_dict, json_file, indent=4)
                if tmpdir is not None and eval_on_format_results:
                    # remove tmp dir when cityscapes evaluation
                    shutil.rmtree(tmpdir)
                curr_res = {}
                reg_ood_valid = metric_dict['metric'].pop('reg_ood_valid', False)
                sl_ood_valid = metric_dict['metric'].pop('sl_ood_valid', False)
                in_dist_valid = metric_dict['metric'].pop('in_dist_valid', False)
                if reg_ood_valid:
                    curr_iter_reg_ood_df = pd.DataFrame(
                        data=[[all_checkpoints_iter[i],
                               round(float(metric_dict["metric"]['max_prob.auroc']), 2),
                               round(float(metric_dict["metric"]['max_prob.aupr']), 2),
                               round(float(metric_dict["metric"]['max_prob.fpr95']), 2),
                               round(float(metric_dict["metric"]['max_logit.auroc']), 2),
                               round(float(metric_dict["metric"]['max_logit.aupr']), 2),
                               round(float(metric_dict["metric"]['max_logit.fpr95']), 2),
                               round(float(metric_dict["metric"]['emp_entropy.auroc']), 2),
                               round(float(metric_dict["metric"]['emp_entropy.aupr']), 2),
                               round(float(metric_dict["metric"]['emp_entropy.fpr95']), 2)]],
                        columns=["epoch",
                                 'max_prob.auroc',
                                 'max_prob.aupr',
                                 'max_prob.fpr95',
                                 'max_logit.auroc',
                                 'max_logit.aupr',
                                 'max_logit.fpr95',
                                 'emp_entropy.auroc',
                                 'emp_entropy.aupr',
                                 'emp_entropy.fpr95'
                                 ])
                    reg_ood_summary = reg_ood_summary.append(curr_iter_reg_ood_df, ignore_index=True)

                    for a in ("max_prob", "max_logit", "emp_entropy"):
                        for b in ("auroc", "aupr", "fpr95"):
                            curr_res[f"{a}.{b}"] = float(metric_dict['metric'][f"{a}.{b}"])
                if sl_ood_valid:
                    curr_iter_sl_ood_df = pd.DataFrame(
                        data=[[all_checkpoints_iter[i],
                               round(float(metric_dict["metric"]['u.auroc']), 2),
                               round(float(metric_dict["metric"]['u.aupr']), 2),
                               round(float(metric_dict["metric"]['u.fpr95']), 2),
                               round(float(metric_dict["metric"]['disonnance.auroc']), 2),
                               round(float(metric_dict["metric"]['disonnance.aupr']), 2),
                               round(float(metric_dict["metric"]['disonnance.fpr95']), 2),
                               round(float(metric_dict["metric"]['dir_entropy.auroc']), 2),
                               round(float(metric_dict["metric"]['dir_entropy.aupr']), 2),
                               round(float(metric_dict["metric"]['dir_entropy.fpr95']), 2)]],
                        columns=["epoch",
                                 'u.auroc',
                                 'u.aupr',
                                 'u.fpr95',
                                 'disonnance.auroc',
                                 'disonnance.aupr',
                                 'disonnance.fpr95',
                                 'dir_entropy.auroc',
                                 'dir_entropy.aupr',
                                 'dir_entropy.fpr95'
                                 ])
                    sl_ood_summary = sl_ood_summary.append(curr_iter_sl_ood_df, ignore_index=True)

                    for a in ("u", "disonnance", "dir_entropy"):
                        for b in ("auroc", "aupr", "fpr95"):
                            curr_res[f"{a}.{b}"] = float(metric_dict['metric'][f"{a}.{b}"])

                if in_dist_valid:
                    curr_res["mIoU"] = float(metric_dict['metric']["mIoU"])
                    curr_res["mAcc"] = float(metric_dict['metric']["mAcc"])
                    curr_res["aAcc"] = float(metric_dict['metric']["aAcc"])

                if args.all:
                    for k, v in curr_res.items():
                        writer.add_scalar(f"test/{k}", v, all_checkpoints_iter[i])

                curr_res["iter"] = all_checkpoints_iter[i]
                ans.append(curr_res)

    print("For latex:")
    # import ipdb; ipdb.set_trace()
    if sl_ood_valid:
        print(
            f"{reg_ood_summary.loc[i, 'max_prob.auroc']}/{reg_ood_summary.loc[i, 'max_logit.auroc']}/{reg_ood_summary.loc[i, 'emp_entropy.auroc']}/{sl_ood_summary.loc[i, 'u.auroc']}/{sl_ood_summary.loc[i, 'disonnance.auroc']}"
            f"&{reg_ood_summary.loc[i, 'max_prob.aupr']}/{reg_ood_summary.loc[i, 'max_logit.aupr']}/{reg_ood_summary.loc[i, 'emp_entropy.aupr']}/{sl_ood_summary.loc[i, 'u.aupr']}/{sl_ood_summary.loc[i, 'disonnance.aupr']}"
            f"&{reg_ood_summary.loc[i, 'max_prob.fpr95']}/{reg_ood_summary.loc[i, 'max_logit.fpr95']}/{reg_ood_summary.loc[i, 'emp_entropy.fpr95']}/{sl_ood_summary.loc[i, 'u.fpr95']}/{sl_ood_summary.loc[i, 'disonnance.fpr95']}")
    else:
        print(
            f"{reg_ood_summary.loc[i, 'max_prob.auroc']}/{reg_ood_summary.loc[i, 'max_logit.auroc']}/{reg_ood_summary.loc[i, 'emp_entropy.auroc']}/-.--/-.--"
            f"&{reg_ood_summary.loc[i, 'max_prob.aupr']}/{reg_ood_summary.loc[i, 'max_logit.aupr']}/{reg_ood_summary.loc[i, 'emp_entropy.aupr']}/-.--/-.--"
            f"&{reg_ood_summary.loc[i, 'max_prob.fpr95']}/{reg_ood_summary.loc[i, 'max_logit.fpr95']}/{reg_ood_summary.loc[i, 'emp_entropy.fpr95']}/-.--/-.--")
    suffixe = re.search(r"(?:[0-9]{14})_(.*)$", args.work_dir).groups()[0]
    with open(os.path.join(args.work_dir, f"test_results_all_{suffixe}.json" if args.all else f"test_results_{suffixe}.json"), "w") as f:
        json.dump(ans, f)
    reg_ood_summary.to_csv(os.path.join(args.work_dir, f'reg_ood_metrics_{suffixe}.csv'))
    sl_ood_summary.to_csv(os.path.join(args.work_dir, f'sl_ood_metrics_{suffixe}.csv'))


if __name__ == '__main__':
    main()
