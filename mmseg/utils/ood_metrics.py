import numpy as np
import sklearn.metrics as sk
from mmcv.utils import print_log
from ood_metrics import fpr_at_95_tpr
import torch
import torch.nn.functional as F
RECALL_LEVEL_DEFAULT = 0.95
np.seterr(invalid='ignore')


# Calculate dissonance of a vector of alphas
def diss(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    bel = evi / s
    cls = np.arange(alpha.shape[1])
    diss = 0
    def Bal(bi, bj): return 1 - torch.abs(bi - bj) / (bi + bj + 1e-8)
    for i in cls:
        score_j_bal = [bel[:, j] * Bal(bel[:, j], bel[:, i]) for j in cls[cls != i]]
        score_j = [bel[:, j] for j in cls[cls != i]]
        diss += bel[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-8)
    return diss


def brierscore(probs, target, reduction='mean'):
    y_one_hot = torch.nn.functional.one_hot(target, num_classes=probs.shape[1])
    squared_diff = torch.sum((y_one_hot - probs) ** 2, axis=1)
    sum_squared_diff = torch.sum(squared_diff)

    if reduction == 'mean':
        return sum_squared_diff / float(2 * probs.shape[0])
    elif reduction == 'sum':
        return sum_squared_diff
    elif reduction == 'none':
        return squared_diff
    else:
        raise KeyError('reduction')


def get_measures(_pos, _neg):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_at_95_tpr(examples, labels)

    return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, ece, logger=None, text="max_prob"):
    print_log(f'OOD metrics computed using: {text}')
    print_log('FPR{:d}: {:.2f}'.format(int(100 * RECALL_LEVEL_DEFAULT), 100 * fpr), logger)
    print_log('AUROC: {:.2f}'.format(100 * auroc), logger)
    print_log('AUPR: {:.2f}'.format(100 * aupr), logger)


def print_measures_with_std(aurocs, auprs, fprs, eces, logger=None, text="max_prob"):
    print_log(f'OOD metrics computed using: {text}')
    print_log('FPR{:d}: {:.2f} +/- {:.2f}'.format(int(100 * RECALL_LEVEL_DEFAULT), 100 * np.mean(fprs), 100 * np.std(fprs)), logger)
    print_log('AUROC: {:.2f} +/- {:.2f}'.format(100 * np.mean(aurocs), 100 * np.std(aurocs)), logger)
    print_log('AUPR: {:.2f} +/- {:.2f}'.format(100 * np.mean(auprs), 100 * np.std(auprs)), logger)
