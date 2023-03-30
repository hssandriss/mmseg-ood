# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings
import torch.nn.functional as F
import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from mmcv.parallel import is_module_wrapper
from mmcv.utils import print_log
import time


def avgfusion(alpha):
    ev = alpha - 1
    comb_ev = ev.mean(0, keepdim=True)
    comb_alpha = comb_ev + 1
    s = comb_alpha.sum(1, keepdim=True)
    u = comb_alpha.size(1) / s
    probs = comb_alpha / (s + 1e-9)
    bel = comb_ev / (s + 1e-9)
    return bel, u, comb_alpha, probs


def wfusion(alpha):
    ev = alpha - 1
    indiv_s = alpha.sum(1, keepdim=True)
    indiv_u = alpha.size(1) / indiv_s
    comb_ev = (ev * (1 - indiv_u)).sum(0, keepdim=True) / (1 - indiv_u).sum(0, keepdim=True)
    comb_alpha = comb_ev + 1
    s = comb_alpha.sum(1, keepdim=True)
    u = comb_alpha.size(1) / s
    probs = comb_alpha / (s + 1e-9)
    bel = comb_ev / (s + 1e-9)
    return bel, u, comb_alpha, probs

def ccfusion(alpha, combinations):
    EPS = 1e-9
    s = alpha.sum(1, keepdim=True)
    ev = alpha - 1
    bel = ev / s
    W = ev.size(1)
    u = W / s
    a = 1 / (ev.size(1) - (ev.size(0) - 1))
    bel_cons_x = bel.min(dim=0, keepdim=True)[0]
    bel_cons = bel_cons_x.sum(1, keepdim=True)
    bel_res_x = bel - bel_cons_x

    u_pre = u.prod(0, keepdim=True)
    bel_comp_x = (bel_res_x * u_pre / (u + EPS)).sum(0, keepdim=True)  # first part of sum [1, 19, 720, 1280]
    u_comp = torch.zeros_like(u_pre)  # Comp on X (the whole interval)

    for comb in combinations:
        if len(set(comb)) == 1:
            # Intersection at x
            bel_comp_x[0, comb[0], :, :] += a**ev.size(0) * bel_res_x[range(ev.size(0)), comb, :, :].prod(0)
            # Union at x and non null intersection
            bel_comp_x[0, comb[0], :, :] += (1 - a**ev.size(0)) * bel_res_x[range(ev.size(0)), comb, :, :].prod(0)
        else:
            # Union at x and null intersection
            prod = bel_res_x[range(ev.size(0)), comb, :, :].prod(0)
            if len(set(comb)) == ev.size(1):
                u_comp[0, 0, :, :] += prod
            else:
                for c in set(comb):
                    bel_comp_x[0, c, :, :] += prod

    bel_comp = bel_comp_x.sum(1, keepdim=True) + u_comp
    nu = (1 - bel_cons - u_pre) / (bel_comp)

    comb_bel = bel_cons_x + nu * bel_comp_x
    comb_u = u_pre + nu * u_comp
    comb_prob = comb_bel + comb_u * (1 / ev.size(1))

    # assert torch.allclose(comb_u + comb_bel.sum(1, keepdim=True), torch.ones_like(comb_u))
    return comb_bel, comb_u, comb_prob


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def plot_mask(mask, file):
    plt.figure()
    sns.heatmap(mask,
                xticklabels=False,
                yticklabels=False, cmap='plasma').get_figure().savefig(file, dpi=300, bbox_inches='tight')
    plt.cla(); plt.clf(); plt.close('all')


def plot_conf(conf, file):
    plt.figure()
    sns.heatmap(
        conf,
        xticklabels=False, yticklabels=False, cmap='plasma').get_figure().savefig(file, dpi=300, bbox_inches='tight')
    plt.cla(); plt.clf(); plt.close('all')


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    fusion_method='af',
                    work_dir="."):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    torchmodel = model
    if is_module_wrapper(torchmodel):
        torchmodel = torchmodel.module

    if hasattr(torchmodel.decode_head, 'density_type') and torchmodel.decode_head.density_type in ("flow", "conditional_flow"):
        torchmodel.decode_head.density_estimation.tdist_to_device()

    def logit2alpha(x):
        # for EDL
        ev = torchmodel.decode_head.loss_decode.logit2evidence(x) + 1
        if torchmodel.decode_head.loss_decode.pow_alpha:
            ev = ev**2
        return ev

    def logit2prob(x):
        # for CE
        if torchmodel.decode_head.loss_decode.use_softplus:
            return F.softplus(x) / F.softplus(x).sum(dim=1, keepdim=True)
        return F.softmax(x, dim=1)

    assert data_loader.batch_size == 1, "TEST SCRIPT ONLY WORKS WITH BATCH SIZE=1"
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler
    time_forward = []
    print_log(f"\n# Total Trainable Parameters: {sum(p.numel() for p in torchmodel.parameters() if p.requires_grad)}")

    print("Using averaging fusion") if fusion_method == 'af' else print("Using weighted fusion")
    var_s = []
    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            # torchmodel.decode_head.dropout.train()
            result, seg_logit = model(return_loss=False, **data)  # returns labels and logits
            torch.cuda.synchronize()
            t1 = time.time()
        time_forward.append(t1 - t0)

        # For testing mcdropout
        # result = []
        # seg_logit = []
        # ts = []
        # for _ in range(20):
        #     with torch.no_grad():
        #         torch.cuda.synchronize()
        #         t0 = time.time()
        #         torchmodel.decode_head.dropout.train()
        #         result_i, seg_logit_i = model(return_loss=False, **data)
        #         torch.cuda.synchronize()
        #         t1 = time.time()
        #     ts.append(t1-t0)
        #     result.append(result_i)
        #     seg_logit.append(seg_logit_i.detach())
        # time_forward.append(np.mean(ts))
        # seg_logit = torch.cat(seg_logit, dim=0)

        n_samples = seg_logit.size(0)
        if n_samples > 1:
            assert fusion_method in ('af', 'wf')
        else:
            fusion_method = 'none'
        seg_logit = seg_logit.detach()

        seg_gt = dataset.get_gt_seg_map_by_idx_and_reduce_zero_label(batch_indices[0])
        if (show or out_dir):
            # produce 3 images
            # gt_seg_map, pred_seg_map, confidence_map
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])

            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]

                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None
                if dataset.reduce_zero_label:
                    ignore_mask = (seg_gt == dataset.ignore_index).astype(np.bool)
                else:
                    ignore_mask = np.zeros_like(seg_gt).astype(np.bool)
                for idx, res in enumerate(result):
                    torchmodel.show_result(
                        img_show,
                        [np.ma.array(res, mask=ignore_mask)],
                        palette=dataset.PALETTE,
                        show=show,
                        out_file=out_file[:-4] + f"_{idx}" + out_file[-4:],
                        opacity=opacity)
                torchmodel.show_result(
                    img_show,
                    [seg_gt],
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file[:-4] + "_gt" + out_file[-4:],
                    opacity=opacity)

                
                if torchmodel.decode_head.loss_decode.loss_name.startswith("loss_edl"):
                    # EDL loss
                    num_cls = seg_logit.shape[1]
                    alpha = logit2alpha(seg_logit)
                    strength = alpha.sum(dim=1, keepdim=True)
                    if n_samples > 1:
                        # Results obtained using averaging fusion
                        _, comb_u, comb_alpha, comb_probs = avgfusion(alpha)
                        comb_strength = comb_alpha.sum(dim=1, keepdim=True)
                        comb_dir_entropy = (torch.lgamma(comb_alpha).sum(1, keepdim=True) - torch.lgamma(comb_strength) -
                                            (num_cls - comb_strength) * torch.digamma(comb_strength) -
                                            ((comb_alpha - 1.0) * torch.digamma(comb_alpha)).sum(1, keepdim=True)).squeeze().cpu().numpy()
                        normalized_comb_dir_entropy = comb_dir_entropy
                       

                        normalized_comb_u = (comb_u - comb_u.min()) / (comb_u.max() - comb_u.min())

                        normalized_comb_u = normalized_comb_u.cpu().squeeze().numpy()
                        normalized_comb_u[ignore_mask] = 0.

                        plot_conf(normalized_comb_u, out_file[: -4] + "_edl_avgfusion_u" + out_file[-4:])
                        normalized_comb_dir_entropy[ignore_mask] = normalized_comb_dir_entropy.min()
                        plot_conf(normalized_comb_dir_entropy, out_file[: -4] + "_edl_avgfusion_dir_entropy" + out_file[-4:])
                        comb_pred = comb_probs.max(1)[1].squeeze().cpu().numpy()  # predictions
                        torchmodel.show_result(
                            img_show,
                            [np.ma.array(comb_pred, mask=ignore_mask)],
                            palette=dataset.PALETTE,
                            show=show,
                            out_file=out_file[:-4] + "_avgfusion_pred_" + out_file[-4:],
                            opacity=opacity)
                        # Results obtained using weighted fusion
                        _, comb_u, comb_alpha, comb_probs = wfusion(alpha)
                        comb_strength = comb_alpha.sum(dim=1, keepdim=True)
                        comb_dir_entropy = (torch.lgamma(comb_alpha).sum(1, keepdim=True) - torch.lgamma(comb_strength) -
                                            (num_cls - comb_strength) * torch.digamma(comb_strength) -
                                            ((comb_alpha - 1.0) * torch.digamma(comb_alpha)).sum(1, keepdim=True)).squeeze().cpu().numpy()
                        normalized_comb_dir_entropy = comb_dir_entropy
                        # normalized_comb_dir_entropy = (comb_dir_entropy - comb_dir_entropy.min()
                        #                                ) / (comb_dir_entropy.max() - comb_dir_entropy.min()+1e-8)
                        normalized_comb_u = (comb_u - comb_u.min()) / (comb_u.max() - comb_u.min())

                        normalized_comb_u = normalized_comb_u.cpu().squeeze().numpy()
                        normalized_comb_u[ignore_mask] = 0.
                        plot_conf(normalized_comb_u, out_file[: -4] + "_edl_wfusion_u" + out_file[-4:])
                        # import ipdb; ipdb.set_trace()
                        normalized_comb_dir_entropy[ignore_mask] = normalized_comb_dir_entropy.min()
                        plot_conf(normalized_comb_dir_entropy, out_file[: -4] + "_edl_wfusion_dir_entropy" + out_file[-4:])

                        comb_pred = comb_probs.max(1)[1].squeeze().cpu().numpy()  # predictions
                        torchmodel.show_result(
                            img_show,
                            [np.ma.array(comb_pred, mask=ignore_mask)],
                            palette=dataset.PALETTE,
                            show=show,
                            out_file=out_file[:-4] + "_wfusion_pred_" + out_file[-4:],
                            opacity=opacity)
                    else:
                        dir_entropy = (torch.lgamma(alpha).sum(1, keepdim=True) - torch.lgamma(strength) -
                                        (num_cls - strength) * torch.digamma(strength) -
                                        ((alpha - 1.0) * torch.digamma(alpha)).sum(1, keepdim=True)).squeeze().cpu().numpy()
                        u = (num_cls / alpha.sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
                        u[ignore_mask] = 0.
                        dir_entropy[ignore_mask] = dir_entropy.min()

                        plot_conf(u, out_file[: -4] + "_edl_u" + out_file[-4:])
                        plot_conf(dir_entropy, out_file[: -4] + "_edl_dir_entropy" + out_file[-4:])
                else:
                    # Softmax crossentropy loss
                    probs = logit2prob(seg_logit)
                    if n_samples > 1:
                        probs_avg = probs.mean(dim=0)
                        plot_conf(probs_avg.max(dim=0)[0].cpu().numpy(), out_file[: -4] + f"_sm_avg_prob" + out_file[-4:])
                        plot_conf(probs.mean(dim=0).max(dim=0)[0].cpu().numpy(), out_file[: -4] + f"_sm_mean" + out_file[-4:])
                        plot_conf(probs.var(dim=0).max(dim=0)[0].cpu().numpy(), out_file[: -4] + f"_sm_var_max_prob" + out_file[-4:])
                        plot_conf(probs.max(dim=1)[0].var(dim=0).cpu().numpy(), out_file[: -4] + f"_sm_max_var_prob" + out_file[-4:])
                        for i in range(n_samples):
                            plot_conf(probs[i, :, :, :].max(dim=0)[0].cpu().numpy(), out_file[: -4] + f"_sm_conf_sample_{i}" + out_file[-4:])
                    else:
                        max_logit = seg_logit.max(dim=1)[0].squeeze().cpu().numpy()
                        max_logit[ignore_mask] = max_logit.max()
                        plot_conf(- max_logit, out_file[: -4] + f"_ml_conf" + out_file[-4:])

                        probs = probs.max(dim=1)[0].squeeze().cpu().numpy()
                        probs[ignore_mask] = probs.max()
                        plot_conf(1 - probs, out_file[: -4] + f"_sm_conf_sample" + out_file[-4:])
                if hasattr(dataset, "ood_indices"):
                    plot_mask((seg_gt == dataset.ood_indices[0]).astype(np.uint8), out_file[:-4] + "_ood_mask" + out_file[-4:])

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # For originally included metrics mIOU
            if torchmodel.decode_head.loss_decode.loss_name.startswith("loss_edl") and n_samples > 1:
                if fusion_method == 'af':
                    _, _, _, probs_f = avgfusion(logit2alpha(seg_logit))
                else:
                    _, _, _, probs_f = wfusion(logit2alpha(seg_logit))
                p = logit2alpha(seg_logit) / logit2alpha(seg_logit).sum(1, keepdim=True)
                var_s.append(p.squeeze().cpu().numpy().var(0, keepdims=True))
                result = [probs_f.max(1)[1].squeeze().cpu().numpy()]  # predictions
            result_seg = dataset.pre_eval(result, indices=batch_indices)[0]
            
            # For added metrics OOD, calibration
            if not torchmodel.decode_head.use_bags:
                if torchmodel.decode_head.loss_decode.loss_name.startswith("loss_edl"):
                    # For EDL probs
                    if n_samples > 1:
                        if fusion_method == 'af':
                            result_oth = dataset.pre_eval_custom_many_samples(
                                seg_logit, seg_gt, "edl", logit_fn=logit2alpha, fusion_fn=avgfusion)
                        else:
                            result_oth = dataset.pre_eval_custom_many_samples(
                                seg_logit, seg_gt, "edl", logit_fn=logit2alpha, fusion_fn=wfusion)
                    else:
                        result_oth = dataset.pre_eval_custom_single_sample(seg_logit, seg_gt, "edl", logit_fn=logit2alpha)
                else:
                    # For softmax probs
                    if n_samples > 1:
                        def logit_fn(x): return logit2prob(x).mean(0, keepdim=True)
                        result_oth = dataset.pre_eval_custom_many_samples(seg_logit, seg_gt, "softmax", logit_fn=logit_fn)

                    else:
                        result_oth = dataset.pre_eval_custom_single_sample(seg_logit, seg_gt, "softmax", logit_fn=logit2prob)
            else:
                raise NotImplementedError
            result = [(result_seg, result_oth)]
            results.extend(result)
            pass
        else:
            results.extend(result)

        batch_size = 1
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    torchmodel = model
    if is_module_wrapper(torchmodel):
        torchmodel = torchmodel.module

    if hasattr(torchmodel.decode_head, 'density_type') and torchmodel.decode_head.density_type in ("flow", "conditional_flow"):
        torchmodel.decode_head.density_estimation.tdist_to_device()

    def logit2alpha(x):
        # for EDL
        ev = torchmodel.decode_head.loss_decode.logit2evidence(x) + 1
        if torchmodel.decode_head.loss_decode.pow_alpha:
            ev = ev**2
        return ev

    def logit2prob(x):
        # for CE
        if torchmodel.decode_head.loss_decode.use_softplus:
            return F.softplus(x) / F.softplus(x).sum(dim=1, keepdim=True)
        return F.softmax(x, dim=1)
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result, seg_logit = model(return_loss=False, rescale=True, **data)

        n_samples = seg_logit.size(0)
        seg_logit = seg_logit.detach()
        seg_gt = dataset.get_gt_seg_map_by_idx_and_reduce_zero_label(batch_indices[0])

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # For originally included metrics mIOU
            result_seg = dataset.pre_eval(result, indices=batch_indices)[0]
            # For added metrics OOD, calibration
            if torchmodel.decode_head.loss_decode.loss_name.startswith("loss_edl"):
                # For EDL probs
                if n_samples > 1:
                    def logit_fn(x): return ccfusion(logit2alpha(x), torchmodel.decode_head.combinations)
                    result_oth = dataset.pre_eval_custom_many_samples(seg_logit, seg_gt, "edl", logit_fn=logit_fn)
                    # def logit_fn(x): return avgfusion(logit2alpha(x))
                    # result_oth = dataset.pre_eval_custom_many_samples(seg_logit, seg_gt, "edl", logit_fn=logit_fn)

                else:
                    result_oth = dataset.pre_eval_custom_single_sample(seg_logit, seg_gt, "edl", logit_fn=logit2alpha)
            else:
                # For softmax probs
                if n_samples > 1:
                    def logit_fn(x): return logit2prob(x).mean(0, keepdim=True)
                    result_oth = dataset.pre_eval_custom_many_samples(seg_logit, seg_gt, "softmax", logit_fn=logit_fn)
                else:
                    result_oth = dataset.pre_eval_custom_single_sample(seg_logit, seg_gt, "softmax", logit_fn=logit2prob)

            result = [(result_seg, result_oth)]
            results.extend(result)
        else:
            results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
