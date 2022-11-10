# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict
from mmseg.models.losses import edl_kld
import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset
from sklearn.utils.validation import assert_all_finite
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations
import torch.nn.functional as F
import torch
from torchmetrics.functional import calibration_error, pearson_corrcoef
from ..utils import brierscore, diss


@DATASETS.register_module()
class CustomDataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/en/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
        gt_seg_map_loader_cfg (dict): build LoadAnnotations to load gt for
            evaluation, load from disk by default. Default: ``dict()``.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=dict(),
                 file_client_args=dict(backend='disk')):
        self.mixed = False
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.gt_seg_map_loader = LoadAnnotations(
            reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient.infer_client(self.file_client_args)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        raise NotImplementedError

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        return results['gt_semantic_seg']

    def get_gt_seg_map_by_idx_and_reduce_zero_label(self, index):
        seg_gt = self.get_gt_seg_map_by_idx(index)
        if self.reduce_zero_label:
            seg_gt[seg_gt == 0] = 255
            seg_gt = seg_gt - 1
            seg_gt[seg_gt == 254] = 255
        return seg_gt

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            yield results['gt_semantic_seg']

    def pre_eval_custom_many_samples(self, seg_logit, seg_gt, logit2prob="softmax", logit_fn=lambda x: x, fusion_fn=lambda x: x):
        NA = np.nan  # value when metric is not used
        seg_logit = seg_logit.cpu()
        num_cls = seg_logit.shape[1]

        # seg_logit_flat = seg_logit.mean(dim=0, keepdim=True).flatten(2, -1).squeeze().permute(1, 0)  # [1, K, W, H] => [WxH, K]

        seg_gt_tensor_flat = torch.from_numpy(seg_gt).type(torch.long).flatten()  # [W, H] => [WxH]
        if self.ignore_index:
            ignore_bg_mask = (seg_gt_tensor_flat == self.ignore_index)  # ignore bg pixels
        else:
            ignore_bg_mask = torch.zeros_like(seg_gt_tensor_flat)

        if logit2prob == "edl":
            bel, u, probs = fusion_fn(logit_fn(seg_logit))
            bel = bel.flatten(2, -1).squeeze(0).permute(1, 0)
            probs = probs.flatten(2, -1).squeeze(0).permute(1, 0)
            u = u.flatten(2, -1).squeeze(0).permute(1, 0)
            seg_u = u.squeeze()
            disonnance = diss(bel)
            seg_max_prob = probs.max(dim=1)[0]
            seg_max_logit = bel.max(dim=1)[0]  # bel and ev are related through b_i = e_i /s
            ########## uncertainty maximization ##########
            # proj_prob = bel + u * (1 / num_cls)
            # um_u = torch.min(num_cls * proj_prob, dim=1, keepdim=True)[0]
            # um_bel = proj_prob - um_u * (1 / num_cls)
            # seg_um_u = um_u.squeeze()
            ##############################################
            # sb = bel.sum(1, keepdim=True)
            # strength = (num_cls / (1 - sb + 1e-16))
            # evidence = bel * strength

            # alpha = evidence + 1
            # seg_um_u = edl_kld(alpha, num_cls).squeeze()
            seg_emp_entropy = - (probs * probs.clip(1e-6, 1).log()).sum(1)
            
            indiv_alphas = logit_fn(seg_logit)
            indiv_s = indiv_alphas.sum(1, keepdim=True)
            indiv_u =  num_cls/ indiv_s
            # import ipdb; ipdb.set_trace()
            seg_um_u = indiv_u.var(0, keepdim=True).flatten(2, -1).squeeze(0).permute(1, 0).squeeze()
            # seg_dir_entropy = (torch.lgamma(alpha).sum(1, keepdim=True) - to  rch.lgamma(strength) -
            #                    (num_cls - strength) * torch.digamma(strength) -
            #                    ((alpha - 1.0) * torch.digamma(alpha)).sum(1, keepdim=True))
            # import ipdb; ipdb.set_trace()
            seg_dir_entropy = torch.full(size=seg_max_prob.shape, fill_value=NA)
            seg_disonnance = disonnance.squeeze()
        else:
            probs = logit_fn(seg_logit)
            probs = probs.flatten(2, -1).squeeze(0).permute(1, 0)
            seg_max_prob = probs.max(dim=1)[0]
            seg_emp_entropy = - (probs * probs.clip(1e-6, 1).log()).sum(1)

            # Both seg_var_sum and seg_inf_gain are disguised under max_logit column
            # https://arxiv.org/abs/1803.08533
            seg_probs = F.softmax(seg_logit, dim=1)
            seg_var_sum = seg_probs.var(dim=0).sum(dim=0).flatten(0, -1)
            seg_mean_emp_entropy = (-seg_probs * seg_probs.clip(1e-6, 1).log()).sum(1, keepdim=True).mean(0, keepdim=True).squeeze().flatten(0, -1)
            seg_inf_gain = seg_emp_entropy - seg_mean_emp_entropy
            seg_u = torch.full(size=seg_max_prob.shape, fill_value=NA)
            seg_dir_entropy = torch.full(size=seg_max_prob.shape, fill_value=NA)
            seg_disonnance = torch.full(size=seg_max_prob.shape, fill_value=NA)
            seg_max_logit = torch.full(size=seg_max_prob.shape, fill_value=NA)

        # Compute OOD metrics for openset
        if hasattr(self, "ood_indices"):

            ood_mask = (seg_gt_tensor_flat == self.ood_indices[0])
            ood_valid = ood_mask.any() and (~ood_mask).any()

            if ood_valid:
                seg_gt_array_flat = seg_gt_tensor_flat.cpu().numpy()
                assert_all_finite(seg_gt_array_flat)

                seg_max_prob_array = seg_max_prob.cpu().numpy()
                assert_all_finite(seg_max_prob_array)
                out_scores_probs, in_scores_probs = self.get_in_out_conf(seg_max_prob_array, seg_gt_array_flat, "max_prob")
                auroc_prob, aupr_prob, fpr_prob = self.evaluate_ood(out_scores_probs, in_scores_probs)
                probs_ood = np.array([auroc_prob, aupr_prob, fpr_prob])

                seg_emp_entropy_array = seg_emp_entropy.cpu().numpy()
                assert_all_finite(seg_emp_entropy_array)
                out_scores_emp_entr, in_scores_emp_entr = self.get_in_out_conf(seg_emp_entropy_array, seg_gt_array_flat, "entropy")
                auroc_emp_entr, aupr_emp_entropy, fpr_emp_entropy = self.evaluate_ood(out_scores_emp_entr, in_scores_emp_entr)
                emp_entr_ood = np.array([auroc_emp_entr, aupr_emp_entropy, fpr_emp_entropy])

                if logit2prob == "edl":
                    seg_disonnance_array = seg_disonnance.cpu().numpy()
                    assert_all_finite(seg_disonnance_array)
                    out_scores_diss, in_scores_diss = self.get_in_out_conf(seg_disonnance_array, seg_gt_array_flat, "dissonance")
                    auroc_diss, aupr_diss, fpr_diss = self.evaluate_ood(out_scores_diss, in_scores_diss)
                    dissonance_ood = np.array([auroc_diss, aupr_diss, fpr_diss])

                    seg_u_array = seg_u.cpu().numpy()
                    assert_all_finite(seg_u_array)
                    out_scores_u, in_scores_u = self.get_in_out_conf(seg_u_array, seg_gt_array_flat, "vacuity")
                    auroc_u, aupr_u, fpr_u = self.evaluate_ood(out_scores_u, in_scores_u)
                    u_ood = np.array([auroc_u, aupr_u, fpr_u])

                    seg_um_u_array = seg_um_u.cpu().numpy()
                    assert_all_finite(seg_um_u_array)
                    out_scores_um_u, in_scores_um_u = self.get_in_out_conf(seg_um_u_array, seg_gt_array_flat, "vacuity")
                    auroc_um_u, aupr_um_u, fpr_um_u = self.evaluate_ood(out_scores_um_u, in_scores_um_u)
                    um_u_ood = np.array([auroc_um_u, aupr_um_u, fpr_um_u])

                    seg_max_logit_array = seg_max_logit.cpu().numpy()
                    assert_all_finite(seg_max_logit_array)
                    out_scores_logit, in_scores_logit = self.get_in_out_conf(seg_max_logit_array, seg_gt_array_flat, "max_logit")
                    auroc_logit, aupr_logit, fpr_logit = self.evaluate_ood(out_scores_logit, in_scores_logit)
                    logit_ood = np.array([auroc_logit, aupr_logit, fpr_logit])

                    # seg_dir_entropy_array = seg_dir_entropy.cpu().numpy()
                    # assert_all_finite(seg_dir_entropy_array)
                    # out_scores_dir_entr, in_scores_dir_entr = self.get_in_out_conf(seg_dir_entropy_array, seg_gt_array_flat, "entropy")
                    # auroc_dir_entr, aupr_dir_entr, fpr_dir_entr = self.evaluate_ood(out_scores_dir_entr, in_scores_dir_entr)
                    # dir_entr_ood = np.array([auroc_dir_entr, aupr_dir_entr, fpr_dir_entr])
                    dir_entr_ood = np.array([NA, NA, NA])
                else:
                    dissonance_ood = np.array([NA, NA, NA])
                    u_ood = np.array([NA, NA, NA])
                    um_u_ood = np.array([NA, NA, NA])
                    dir_entr_ood = np.array([NA, NA, NA])

                    # seg_inf_gain_array = seg_var_sum.cpu().numpy()
                    # assert_all_finite(seg_inf_gain_array)
                    # out_scores_inf_gain, in_scores_inf_gain = self.get_in_out_conf(seg_inf_gain_array, seg_gt_array_flat, "entropy")
                    # auroc_inf_gain, aupr_inf_gain, fpr_inf_gain = self.evaluate_ood(out_scores_inf_gain, in_scores_inf_gain)
                    # logit_ood = np.array([auroc_inf_gain, aupr_inf_gain, fpr_inf_gain])

                    seg_var_sum_array = seg_var_sum.cpu().numpy()
                    assert_all_finite(seg_var_sum_array)
                    out_scores_var_sum, in_scores_var_sum = self.get_in_out_conf(seg_var_sum_array, seg_gt_array_flat, "entropy")
                    auroc_var_sum, aupr_var_sum, fpr_var_sum = self.evaluate_ood(out_scores_var_sum, in_scores_var_sum)
                    logit_ood = np.array([auroc_var_sum, aupr_var_sum, fpr_var_sum])

                corr_max_prob_u = np.array([pearson_corrcoef(seg_max_prob, seg_u)])
                ood_metrics = (np.hstack((probs_ood, logit_ood, emp_entr_ood, u_ood, um_u_ood, dissonance_ood, dir_entr_ood, corr_max_prob_u)), True)
            else:
                ood_metrics = (np.array([NA for _ in range(7 * 3 + 1)]), False)
        else:
            # Puts nans otherwise
            ood_metrics = (np.array([NA for _ in range(7 * 3 + 1)]), False)

        # Calibration/Confidence metrics for closed set
        if not hasattr(self, "ood_indices") or self.mixed:
            if self.ignore_index:
                # filtered out ignored indices
                seg_gt_tensor_flat_no_bg = seg_gt_tensor_flat[~ignore_bg_mask]
                probs_no_bg = probs[~ignore_bg_mask, :]
            if self.mixed:
                # filtered out ood indices
                ood_masker = self.get_ood_masker(seg_gt_tensor_flat_no_bg)
                probs_no_bg = probs_no_bg[ood_masker]
                seg_gt_tensor_flat_no_bg = seg_gt_tensor_flat_no_bg[ood_masker]

            nll = F.nll_loss(probs_no_bg.log(), seg_gt_tensor_flat_no_bg, reduction='mean').item()
            ece_l1 = calibration_error(probs_no_bg, seg_gt_tensor_flat_no_bg, norm='l1', ).item()
            ece_l2 = calibration_error(probs_no_bg, seg_gt_tensor_flat_no_bg, norm='l2').item()
            brier = brierscore(probs_no_bg, seg_gt_tensor_flat_no_bg, reduction="mean").item()
            if logit2prob == "edl":
                corr_max_prob_u = pearson_corrcoef(seg_max_prob, seg_u).item()
            else:
                corr_max_prob_u = np.nan
            calib_metrics = np.array([nll, ece_l1, ece_l2, brier, corr_max_prob_u])

            per_cls_prob = [NA for _ in range(num_cls)]
            per_cls_u = [NA for _ in range(num_cls)]
            per_cls_strength = [NA for _ in range(num_cls)]
            pre_cls_disonnance = [NA for _ in range(num_cls)]

            for c in range(num_cls):
                mask_cls = (seg_gt_tensor_flat == c)
                if mask_cls.any():
                    per_cls_prob[c] = probs[mask_cls, c].sum().item()
                    if logit2prob == "edl":
                        per_cls_u[c] = u[mask_cls].sum().item()
                        per_cls_strength[c] = strength[mask_cls].sum().item()
                        pre_cls_disonnance[c] = disonnance[mask_cls].sum().item()
            per_cls_prob = np.array(per_cls_prob)
            per_cls_u = np.array(per_cls_u)
            per_cls_strength = np.array(per_cls_strength)
            pre_cls_disonnance = np.array(pre_cls_disonnance)
            per_cls_conf_metrics = (per_cls_prob, per_cls_u, per_cls_strength, pre_cls_disonnance)
        else:
            calib_metrics = np.array([NA for _ in range(5)])
            per_cls_prob = np.array([NA for _ in range(num_cls)])
            per_cls_u = np.array([NA for _ in range(num_cls)])
            pre_cls_disonnance = np.array([NA for _ in range(num_cls)])
            per_cls_strength = np.array([NA for _ in range(num_cls)])
            per_cls_conf_metrics = (per_cls_prob, per_cls_u, per_cls_strength, pre_cls_disonnance)
        return (ood_metrics, calib_metrics, per_cls_conf_metrics)

    def pre_eval_custom_single_sample(self, seg_logit, seg_gt, logit2prob="softmax", logit_fn=lambda x: x):
        NA = np.nan  # value when metric is not used
        seg_logit = seg_logit.cpu()
        num_cls = seg_logit.shape[1]

        seg_gt_tensor_flat = torch.from_numpy(seg_gt).type(torch.long).flatten()  # [W, H] => [WxH]
        seg_logit_flat = seg_logit.mean(dim=0, keepdim=True).flatten(2, -1).squeeze().permute(1, 0)  # [1, K, W, H] => [WxH, K]
        if self.ignore_index:
            ignore_bg_mask = (seg_gt_tensor_flat == self.ignore_index)  # ignore bg pixels
        else:
            ignore_bg_mask = torch.zeros_like(seg_gt_tensor_flat)

        if logit2prob == "edl":
            alpha = logit_fn(seg_logit_flat)
            strength = alpha.sum(dim=1, keepdim=True)
            u = num_cls / strength

            probs = alpha / strength
            seg_max_prob = probs.max(dim=1)[0]

            seg_max_logit = seg_logit_flat.max(dim=1)[0]
            seg_u = u.squeeze()
            
            ########## uncertainty maximization ##########
            evi = alpha - 1
            bel = evi / strength
            disonnance = diss(bel)
            # proj_prob = bel + u * (1 / num_cls)
            # um_u = torch.min(num_cls * proj_prob, dim=1, keepdim=True)[0]
            # um_bel = proj_prob - um_u * (1 / num_cls)
            # seg_um_u = um_u.squeeze()
            seg_um_u = bel.max(dim=1)[0]

            ##############################################
            seg_emp_entropy = - (probs * probs.log()).sum(1)
            seg_dir_entropy = (torch.lgamma(alpha).sum(1, keepdim=True) - torch.lgamma(strength) -
                               (num_cls - strength) * torch.digamma(strength) -
                               ((alpha - 1.0) * torch.digamma(alpha)).sum(1, keepdim=True))
            seg_disonnance = disonnance.squeeze()
        else:
            probs = logit_fn(seg_logit_flat)
            seg_max_prob = probs.max(dim=1)[0]
            seg_max_logit = seg_logit_flat.max(dim=1)[0]
            seg_emp_entropy = - (probs * probs.log()).sum(1)
            seg_u = torch.full(size=seg_max_prob.shape, fill_value=NA)
            seg_dir_entropy = torch.full(size=seg_max_prob.shape, fill_value=NA)
            seg_disonnance = torch.full(size=seg_max_prob.shape, fill_value=NA)

        # Compute OOD metrics for openset
        if hasattr(self, "ood_indices"):
            ood_mask = (seg_gt_tensor_flat == self.ood_indices[0])
            ood_valid = ood_mask.any() and (~ood_mask).any()
            if ood_valid:
                seg_gt_array_flat = seg_gt_tensor_flat.cpu().numpy()
                assert_all_finite(seg_gt_array_flat)

                seg_max_prob_array = seg_max_prob.cpu().numpy()
                assert_all_finite(seg_max_prob_array)
                out_scores_probs, in_scores_probs = self.get_in_out_conf(seg_max_prob_array, seg_gt_array_flat, "max_prob")
                auroc_prob, aupr_prob, fpr_prob = self.evaluate_ood(out_scores_probs, in_scores_probs)
                probs_ood = np.array([auroc_prob, aupr_prob, fpr_prob])

                seg_max_logit_array = seg_max_logit.cpu().numpy()
                assert_all_finite(seg_max_logit_array)
                out_scores_logit, in_scores_logit = self.get_in_out_conf(seg_max_logit_array, seg_gt_array_flat, "max_logit")
                auroc_logit, aupr_logit, fpr_logit = self.evaluate_ood(out_scores_logit, in_scores_logit)
                logit_ood = np.array([auroc_logit, aupr_logit, fpr_logit])

                seg_emp_entropy_array = seg_emp_entropy.cpu().numpy()
                assert_all_finite(seg_max_prob_array)
                out_scores_emp_entr, in_scores_emp_entr = self.get_in_out_conf(seg_emp_entropy_array, seg_gt_array_flat, "entropy")
                auroc_emp_entr, aupr_emp_entropy, fpr_emp_entropy = self.evaluate_ood(out_scores_emp_entr, in_scores_emp_entr)
                emp_entr_ood = np.array([auroc_emp_entr, aupr_emp_entropy, fpr_emp_entropy])

                if logit2prob == "edl":
                    seg_disonnance_array = seg_disonnance.cpu().numpy()
                    assert_all_finite(seg_disonnance_array)
                    out_scores_diss, in_scores_diss = self.get_in_out_conf(seg_disonnance_array, seg_gt_array_flat, "dissonance")
                    auroc_diss, aupr_diss, fpr_diss = self.evaluate_ood(out_scores_diss, in_scores_diss)
                    dissonance_ood = np.array([auroc_diss, aupr_diss, fpr_diss])

                    seg_u_array = seg_u.cpu().numpy()
                    assert_all_finite(seg_u_array)
                    out_scores_u, in_scores_u = self.get_in_out_conf(seg_u_array, seg_gt_array_flat, "vacuity")
                    auroc_u, aupr_u, fpr_u = self.evaluate_ood(out_scores_u, in_scores_u)
                    u_ood = np.array([auroc_u, aupr_u, fpr_u])

                    seg_um_u_array = seg_um_u.cpu().numpy()
                    assert_all_finite(seg_um_u_array)
                    out_scores_um_u, in_scores_um_u = self.get_in_out_conf(seg_um_u_array, seg_gt_array_flat, "vacuity")
                    auroc_um_u, aupr_um_u, fpr_um_u = self.evaluate_ood(out_scores_um_u, in_scores_um_u)
                    um_u_ood = np.array([auroc_um_u, aupr_um_u, fpr_um_u])

                    seg_dir_entropy_array = seg_dir_entropy.cpu().numpy()
                    assert_all_finite(seg_dir_entropy_array)
                    out_scores_dir_entr, in_scores_dir_entr = self.get_in_out_conf(seg_dir_entropy_array, seg_gt_array_flat, "entropy")
                    auroc_dir_entr, aupr_dir_entr, fpr_dir_entr = self.evaluate_ood(out_scores_dir_entr, in_scores_dir_entr)
                    dir_entr_ood = np.array([auroc_dir_entr, aupr_dir_entr, fpr_dir_entr])
                else:
                    dissonance_ood = np.array([NA, NA, NA])
                    u_ood = np.array([NA, NA, NA])
                    um_u_ood = np.array([NA, NA, NA])
                    dir_entr_ood = np.array([NA, NA, NA])

                corr_max_prob_u = np.array([pearson_corrcoef(seg_max_prob, seg_u)])
                ood_metrics = (np.hstack((probs_ood, logit_ood, emp_entr_ood, u_ood, um_u_ood, dissonance_ood, dir_entr_ood, corr_max_prob_u)), True)
            else:
                ood_metrics = (np.array([NA for _ in range(7 * 3 + 1)]), False)
        else:
            # Puts nans otherwise
            ood_metrics = (np.array([NA for _ in range(7 * 3 + 1)]), False)

        # Calibration/Confidence metrics for closed set
        if not hasattr(self, "ood_indices") or self.mixed:
            if self.ignore_index:
                # filtered out ignored indices
                seg_gt_tensor_flat_no_bg = seg_gt_tensor_flat[~ignore_bg_mask]
                probs_no_bg = probs[~ignore_bg_mask, :]
            if self.mixed:
                # filtered out ood indices
                ood_masker = self.get_ood_masker(seg_gt_tensor_flat_no_bg)
                probs_no_bg = probs_no_bg[ood_masker]
                seg_gt_tensor_flat_no_bg = seg_gt_tensor_flat_no_bg[ood_masker]

            nll = F.nll_loss(probs_no_bg.log(), seg_gt_tensor_flat_no_bg, reduction='mean').item()
            ece_l1 = calibration_error(probs_no_bg, seg_gt_tensor_flat_no_bg, norm='l1', ).item()
            ece_l2 = calibration_error(probs_no_bg, seg_gt_tensor_flat_no_bg, norm='l2').item()
            brier = brierscore(probs_no_bg, seg_gt_tensor_flat_no_bg, reduction="mean").item()
            if logit2prob == "edl":
                corr_max_prob_u = pearson_corrcoef(seg_max_prob, seg_u).item()
            else:
                corr_max_prob_u = np.nan
            calib_metrics = np.array([nll, ece_l1, ece_l2, brier, corr_max_prob_u])

            per_cls_prob = [NA for _ in range(num_cls)]
            per_cls_u = [NA for _ in range(num_cls)]
            per_cls_strength = [NA for _ in range(num_cls)]
            pre_cls_disonnance = [NA for _ in range(num_cls)]

            for c in range(num_cls):
                mask_cls = (seg_gt_tensor_flat == c)
                if mask_cls.any():
                    per_cls_prob[c] = probs[mask_cls, c].sum().item()
                    if logit2prob == "edl":
                        per_cls_u[c] = u[mask_cls].sum().item()
                        per_cls_strength[c] = strength[mask_cls].sum().item()
                        pre_cls_disonnance[c] = disonnance[mask_cls].sum().item()
            per_cls_prob = np.array(per_cls_prob)
            per_cls_u = np.array(per_cls_u)
            per_cls_strength = np.array(per_cls_strength)
            pre_cls_disonnance = np.array(pre_cls_disonnance)
            per_cls_conf_metrics = (per_cls_prob, per_cls_u, per_cls_strength, pre_cls_disonnance)
        else:
            calib_metrics = np.array([NA for _ in range(5)])
            per_cls_prob = np.array([NA for _ in range(num_cls)])
            per_cls_u = np.array([NA for _ in range(num_cls)])
            pre_cls_disonnance = np.array([NA for _ in range(num_cls)])
            per_cls_strength = np.array([NA for _ in range(num_cls)])
            per_cls_conf_metrics = (per_cls_prob, per_cls_u, per_cls_strength, pre_cls_disonnance)

        return (ood_metrics, calib_metrics, per_cls_conf_metrics)

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit, shape (N, K, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction, area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)

            # Mask ood examples
            if hasattr(self, "ood_indices"):
                ood_masker = self.get_ood_masker(seg_map)
                seg_map = seg_map[ood_masker]
                pred = pred[ood_masker]

            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.CLASSES),
                    self.ignore_index,
                    # as the label map has already been applied and zero label
                    # has already been reduced by get_gt_seg_map_by_idx() i.e.
                    # LoadAnnotations.__call__(), these operations should not
                    # be duplicated. See the following issues/PRs:
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # https://github.com/open-mmlab/mmsegmentation/pull/1417
                    # https://github.com/open-mmlab/mmsegmentation/pull/2504
                    # for more details
                    label_map=dict(),
                    reduce_zero_label=False))

        return pre_eval_results

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = 255
                else:
                    self.label_map[i] = class_names.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != 255:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                # Get random state before set seed, and restore
                # random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU', 'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        sl_ood_valid = True
        reg_ood_valid = True
        in_dist_valid = False if self.__class__.__name__ == 'RoadAnomalyDataset' else True
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=False)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        default_metrics = ('aAcc', 'IoU', 'Acc', 'Fscore', 'Precision', 'Recall', 'Dice', '')
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            if ret_metric in default_metrics  # percentage metrics
            else np.round(np.nanmean(ret_metric_value), 2)  # other metrics
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics.pop('aNll', None)
        ret_metrics.pop('aEce1', None)
        ret_metrics.pop('aEce2', None)
        ret_metrics.pop('aBrierScore', None)
        ret_metrics.pop('aBrierScore', None)
        ret_metrics.pop("aCorrMaxprobU", None)
        regular_ood_metrics = [f"{a}.{b}" for a in ("max_prob", "max_logit", "emp_entropy") for b in ("auroc", "aupr", "fpr95")]
        sl_ood_metrics = [f"{a}.{b}" for a in ("u", "um_u", "disonnance", "dir_entropy")
                          for b in ("auroc", "aupr", "fpr95")] + ["ood_corr_max_prob_u"]
        # remove ood metrics ret_metrics_summary
        for k in regular_ood_metrics:
            ret_metrics_summary.pop(k, None)
        regular_ood_metrics_summary = OrderedDict({ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                                                   for ret_metric, ret_metric_value in ret_metrics.items() if ret_metric in regular_ood_metrics})
        for ret_metric in regular_ood_metrics:
            ret_metrics.pop(ret_metric, None)

        for k in sl_ood_metrics:
            ret_metrics_summary.pop(k, None)

        sl_ood_metrics_summary = OrderedDict({ret_metric: np.round(np.nanmean(ret_metric_value), 2)
                                             for ret_metric, ret_metric_value in ret_metrics.items() if ret_metric in sl_ood_metrics})
        for ret_metric in sl_ood_metrics:
            ret_metrics.pop(ret_metric, None)

        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            if ret_metric in ('Acc', 'IoU')
            else np.round(ret_metric_value, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        # valid update
        # in_dist_valid = all([not np.isnan(v) for k, v in ret_metrics_summary.items() if k in default_metrics])
        reg_ood_valid = any([not np.isnan(v) for v in regular_ood_metrics_summary.values()]) and len(regular_ood_metrics_summary) > 0
        sl_ood_valid = any([not np.isnan(v) for v in sl_ood_metrics_summary.values()]) and len(sl_ood_metrics_summary) > 0

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key in ('aAcc', 'aNll', 'aEce1', 'aEce2', 'aBrierScore', 'aCorrMaxprobU'):
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])
        regular_ood_table_data = PrettyTable()
        for key, val in regular_ood_metrics_summary.items():
            regular_ood_table_data.add_column(key, [val])
        sl_ood_table_data = PrettyTable()
        for key, val in sl_ood_metrics_summary.items():
            sl_ood_table_data.add_column(key, [val])

        eval_results['in_dist_valid'] = False
        eval_results['sl_ood_valid'] = False
        eval_results['reg_ood_valid'] = False
        if in_dist_valid:
            print_log('\n' + 'Per class results:', logger)
            print_log('\n' + class_table_data.get_string(), logger=logger)
            print_log('\n' + 'Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

            for key, value in ret_metrics_summary.items():
                if key in ('aAcc', 'aNll', 'aEce1', 'aEce2', 'aBrierScore', 'aCorrMaxprobU'):
                    eval_results[key] = value
                else:
                    eval_results['m' + key] = value

            ret_metrics_class.pop('Class', None)
            for key, value in ret_metrics_class.items():
                eval_results.update({
                    key + '.' + str(name): value[idx]
                    for idx, name in enumerate(class_names)
                })
            eval_results['in_dist_valid'] = True
        if reg_ood_valid:
            print_log('\n' + 'Regular OOD:', logger)
            if len(regular_ood_metrics_summary):
                print_log('\n' + regular_ood_table_data.get_string(), logger=logger)
            else:
                print_log('\n' + "No image w/ OOD objects or all images have just OOD objects", logger)
            for key, value in regular_ood_metrics_summary.items():
                eval_results[key] = value
            eval_results['reg_ood_valid'] = True
        if sl_ood_valid:
            print_log('\n' + 'SL OOD:', logger)
            if len(sl_ood_metrics_summary):
                print_log('\n' + sl_ood_table_data.get_string(), logger=logger)
            else:
                print_log('\n' + "No image w/ OOD objects or all images have just OOD objects", logger)
            for key, value in sl_ood_metrics_summary.items():
                eval_results[key] = value
            eval_results['sl_ood_valid'] = True
        return eval_results
