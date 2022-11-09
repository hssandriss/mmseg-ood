import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from ...utils import diss
import torchmetrics
from mmcv.utils import print_log
import numpy as np
EPS = 1e-9


def relu_evidence(logits):
    # This function to generate evidence is used for the first example
    return F.relu(logits)


def elu_evidence(logits):
    # This function to generate evidence is used for the first example
    return F.elu(logits) + 1


def sigmoid_evidence(logits):
    # This function to generate evidence is used for the first example
    max_evidence = 1e3
    shift = 35
    slope = .1
    return torch.sigmoid(slope * (logits - shift)) * max_evidence


def exp_evidence(logits):
    # This one usually works better and used for the second and third examples
    # For general settings and different datasets, you may try this one first
    return torch.exp(torch.clip(logits / 10, -20., 20.))


def exp_0_evidence(logits):
    # This one usually works better and used for the second and third examples
    # For general settings and different datasets, you may try this one first
    b = logits.max().detach()
    if b > torch.tensor(torch.finfo(torch.float32).max).log():  # 88.72
        import ipdb; ipdb.set_trace()
    return torch.exp(logits - b) * torch.exp(b)


def exp_1_evidence(logits):
    # This one usually works better and used for the second and third examples
    # For general settings and different datasets, you may try this one first
    return 1. / (torch.exp(-logits) + EPS)


def exp_2_evidence(logits):
    # This one usually works better and used for the second and third examples
    # For general settings and different datasets, you may try this one first
    mask_pos = (logits >= 0)
    mask_neg = ~mask_pos
    ans = torch.zeros_like(logits)
    ans[mask_pos] = 1 / (EPS + torch.exp(-logits[mask_pos]))
    ans[mask_neg] = torch.exp(logits[mask_neg])
    return ans


def softplus_evidence(logits):
    # This one is another alternative and
    # usually behaves better than the relu_evidence
    return F.softplus(logits)


def mse_edl_loss(one_hot_gt, alpha, num_classes):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / strength
    # L_err
    A = torch.sum((one_hot_gt - prob)**2, dim=1, keepdim=True)
    # L_var
    B = torch.sum(alpha * (strength - alpha) / (strength * strength * (strength + 1)), dim=1, keepdim=True)
    # L_kl
    # alpha_kl = (alpha - 1) * (1 - one_hot_gt) + 1
    alpha_kl = alpha * (1 - one_hot_gt) + one_hot_gt
    C = KL(alpha_kl, num_classes)

    # _, pred_cls = torch.max(alpha / strength, 1, keepdim=True)
    # _, target = torch.max(one_hot_gt, 1, keepdim=True)
    # accurate_match = torch.eq(pred_cls, target).float()
    # C = (1 - accurate_match) * KL(alpha_kl, num_classes)

    # L_EUC
    D, E = EUC(alpha, one_hot_gt, num_classes)

    return A, B, C, D, E


def ce_edl_loss(one_hot_gt, alpha, num_classes, func, bis=False, with_var=False, reg=False):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    # L_err
    A = torch.sum(one_hot_gt * (func(strength) - func(alpha)), axis=1, keepdims=True)
    B = torch.zeros_like(A.detach())
    if bis:
        B = torch.sum((1 - one_hot_gt) * (func(strength) - func(strength - alpha)), axis=1, keepdims=True)
        # A = A + A_
    if with_var:
        fake_alphas = alpha * (1 - one_hot_gt) + one_hot_gt
        fake_strength = fake_alphas.sum(dim=1, keepdim=True)
        B = torch.sum(
            (fake_alphas * (fake_strength - fake_alphas)) / (fake_strength * fake_strength * (fake_strength + 1)),
            dim=1, keepdim=True)  # fake var
        B = - 0.1 * B.log()  # increase var
    if reg:
        # Larger coef cause difficulty to learn some classes
        B = 0.001 * (strength - (one_hot_gt * alpha).sum(dim=1, keepdims=True))

    # L_kl
    # alpha_kl = (alpha - 1) * (1 - one_hot_gt) + 1
    alpha_kl = alpha * (1 - one_hot_gt) + one_hot_gt
    C = KL(alpha_kl, num_classes)

    # _, pred_cls = torch.max(alpha / strength, 1, keepdim=True)
    # _, target = torch.max(one_hot_gt, 1, keepdim=True)
    # accurate_match = torch.eq(pred_cls, target).float()
    # C = (1 - accurate_match) * KL(alpha_kl, num_classes)

    # L_EUC
    D, E = EUC(alpha, one_hot_gt, num_classes)

    return A, B, C, D, E


def KL(alpha, num_classes):
    beta = torch.ones((1, num_classes, 1, 1), dtype=torch.float32, device=alpha.device)  # uncertain dir

    strength_alpha = torch.sum(alpha, dim=1, keepdim=True)
    strength_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(strength_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(strength_beta)

    dg0 = torch.digamma(strength_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def EUC(alpha, one_hot_gt, num_classes):
    strength = torch.sum(alpha, dim=1, keepdim=True)
    u = num_classes / strength
    _, target = torch.max(one_hot_gt, 1, keepdim=True)
    max_prob, pred_cls = torch.max(alpha / strength, 1, keepdim=True)

    accurate_match = torch.eq(pred_cls, target).float()
    acc_uncertain = - max_prob * torch.log(1 - u + EPS)
    inacc_certain = - (1 - max_prob) * torch.log(u + EPS)
    return accurate_match * acc_uncertain, (1 - accurate_match) * inacc_certain


def lam(epoch_num, total_epochs, annealing_start, annealing_step, annealing_method, annealing_from):
    if annealing_method == 'step':
        annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(
            max(epoch_num - annealing_from, 0) / annealing_step, dtype=torch.float32)) * 0.1
    if annealing_method == 'step_a':
        annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(
            epoch_num / (total_epochs / 2), dtype=torch.float32)) * 0.1
    elif annealing_method == 'exp':
        annealing_coef = .1 * torch.min(
            annealing_start * torch.exp(-torch.log(annealing_start) / (annealing_step - 1) * epoch_num),
            torch.tensor(1.0, dtype=torch.float32))
    elif annealing_method == 'zero':
        annealing_coef = torch.tensor(0., dtype=torch.float32)
    elif annealing_method == 'one':
        annealing_coef = torch.tensor(1., dtype=torch.float32)
    elif annealing_method == 'constant':
        annealing_coef = torch.tensor(.1, dtype=torch.float32)
    else:
        raise NotImplementedError
    return annealing_coef


@ LOSSES.register_module
class EDLLoss(nn.Module):

    def __init__(self, num_classes, loss_variant="mse", annealing_step=10, annealing_method="zero", annealing_from=1, total_epochs=70,
                 annealing_start=0.01, logit2evidence="exp", regularization="kld", reduction="mean", loss_weight=1.0, pow_alpha=False,
                 avg_non_ignore=True, loss_name='loss_edl'):
        super(EDLLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.avg_non_ignore = avg_non_ignore
        if logit2evidence == "exp":
            self.logit2evidence = exp_evidence
        elif logit2evidence == "exp_0":
            self.logit2evidence = exp_0_evidence
        elif logit2evidence == "exp_1":
            self.logit2evidence = exp_1_evidence
        elif logit2evidence == "exp_2":
            self.logit2evidence = exp_2_evidence
        elif logit2evidence == "sig":
            self.logit2evidence = sigmoid_evidence
        elif logit2evidence == "softplus":
            self.logit2evidence = softplus_evidence
        elif logit2evidence == "relu":
            self.logit2evidence = relu_evidence
        elif logit2evidence == "elu":
            self.logit2evidence = elu_evidence
        else:
            raise KeyError(logit2evidence)
        self.regularization = regularization
        self.annealing_step = annealing_step
        self.annealing_from = annealing_from
        self.annealing_method = annealing_method
        self.annealing_start = torch.tensor(annealing_start, dtype=torch.float32)

        self.epoch_num = 0
        self.total_epochs = total_epochs
        # self.temp = np.linspace(10, 0.1, self.total_epochs).tolist()
        # import ipdb; ipdb.set_trace()
        self.lam_schedule = []
        for epoch in range(self.total_epochs):
            # lam(epoch_num, total_epochs, annealing_start, annealing_step, annealing_method, annealing_from)
            self.lam_schedule.append(lam(epoch, self.total_epochs, self.annealing_start,
                                     self.annealing_step, self.annealing_method, self.annealing_from))
        # if self.annealing_method != 'zero':
        #     assert self.lam_schedule[-1].allclose(torch.tensor(1.)), "Please check you schedule!"
        self.pow_alpha = pow_alpha
        self.loss_name = "_".join([loss_name, loss_variant])
        # for logging
        self.last_A = 0
        self.last_B = 0
        self.last_C = 0
        self.last_D = 0
        self.last_E = 0
        self.epoch_num_ = 0
        self.epoch_nums = []
        self.iter_cnt = 0

        print_log(', '.join([f"{i}: {self.lam_schedule[i]:.4f}" for i in range(self.total_epochs)]))
        # print_log(', '.join([f"{i}: {self.temp[i]:.4f}" for i in range(self.total_epochs)]))

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        # print_()log(f"Epoch ---> {self.epoch_num}/{self.total_epochs}")
        assert reduction_override in (None, 'none', 'mean', 'sum')
        if self.epoch_num == self.epoch_num_ + 1 and self.epoch_num > 0:
            if np.mean(self.epoch_nums) % 1 != 0:
                import ipdb; ipdb.set_trace()
            # print_log(f"Temp: {self.temp[self.epoch_num_]}")
            # print_log(f"Lam: {self.lam_schedule[self.epoch_num_]}")
            # print_log(f"Epoch {self.epoch_num_} is done with {self.iter_cnt} iterations")
            self.iter_cnt = 0
            self.epoch_nums = []
        reduction = (reduction_override if reduction_override else self.reduction)
        evidence = self.logit2evidence(pred)
        if (evidence != evidence).any():
            # detecting inf or nans
            import ipdb; ipdb.set_trace()
        # tempering ev
        # evidence = evidence / self.temp[self.epoch_num]

        alpha = (evidence + 1)**2 if self.pow_alpha else evidence + 1

        target_expanded = target.data.unsqueeze(1).clone()
        mask_ignore = (target_expanded == 255)
        target_expanded[mask_ignore] = 0
        one_hot_gt = torch.zeros_like(pred, dtype=torch.uint8).scatter_(1, target_expanded, 1)
        if self.loss_name.endswith("mse"):  # Eq. 5 MSE
            A, B, C, D, E = mse_edl_loss(one_hot_gt, alpha, self.num_classes)
            if self.regularization == 'kld':
                loss = A + B + self.lam_schedule[self.epoch_num] * C
            elif self.regularization == 'euc':
                # D: acc_uncertain, E: inacc_certain
                loss = A + B + self.lam_schedule[self.epoch_num] * D + (1. - self.lam_schedule[self.epoch_num]) * E
            elif self.regularization == 'none':
                loss = A + B
            else:
                raise NotImplementedError
        elif self.loss_name.endswith("ce"):  # Eq. 4 CrossEntropy
            A, B, C, D, E = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.digamma)
            if self.regularization == 'kld':
                loss = A + B + self.lam_schedule[self.epoch_num] * C
            elif self.regularization == 'euc':
                # D: acc_uncertain, E: inacc_certain
                loss = A + B + self.lam_schedule[self.epoch_num] * D + (1. - self.lam_schedule[self.epoch_num]) * E
            elif self.regularization == 'none':
                loss = A + B
            else:
                raise NotImplementedError
        elif self.loss_name.endswith("mll"):  # Eq. 3 Maximum Likelihood Type II
            A, B, C, D, E = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.log)
            if self.regularization == 'kld':
                loss = A + B + self.lam_schedule[self.epoch_num] * C
            elif self.regularization == 'euc':
                # D: acc_uncertain, E: inacc_certain
                loss = A + B + self.lam_schedule[self.epoch_num] * D + (1. - self.lam_schedule[self.epoch_num]) * E
            elif self.regularization == 'none':
                loss = A + B
        elif self.loss_name.endswith("mll_bis"):  # Eq. 3 Maximum Likelihood Type II
            A, B, C, D, E = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.log, bis=True)
            if self.regularization == 'kld':
                loss = A + B + self.lam_schedule[self.epoch_num] * C
            elif self.regularization == 'euc':
                # D: acc_uncertain, E: inacc_certain
                loss = A + B + self.lam_schedule[self.epoch_num] * D + (1. - self.lam_schedule[self.epoch_num]) * E
            elif self.regularization == 'none':
                loss = A + B
        elif self.loss_name.endswith("mll_var"):  # Eq. 3 Maximum Likelihood Type II
            A, B, C, D, E = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.log, bis=False, with_var=True)
            if self.regularization == 'kld':
                loss = A + B + self.lam_schedule[self.epoch_num] * C
            elif self.regularization == 'euc':
                # D: acc_uncertain, E: inacc_certain
                loss = A + B + self.lam_schedule[self.epoch_num] * D + (1. - self.lam_schedule[self.epoch_num]) * E
            elif self.regularization == 'none':
                loss = A + B
        elif self.loss_name.endswith("mll_reg"):  # Eq. 3 Maximum Likelihood Type II
            A, B, C, D, E = ce_edl_loss(one_hot_gt, alpha, self.num_classes, func=torch.log, bis=False, with_var=False, reg=True)
            if self.regularization == 'kld':
                loss = A + B + self.lam_schedule[self.epoch_num] * C
            elif self.regularization == 'euc':
                # D: acc_uncertain, E: inacc_certain
                loss = A + B + self.lam_schedule[self.epoch_num] * D + (1. - self.lam_schedule[self.epoch_num]) * E
            elif self.regularization == 'none':
                loss = A + B

        else:
            raise NotImplementedError

        if ignore_index:
            loss = torch.where(mask_ignore, torch.zeros_like(loss), loss)

        avg_factor = target.numel() - (target == ignore_index).sum().item()
        if reduction == 'mean':
            loss_cls = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss_cls = loss.sum()
        else:
            loss_cls = loss

        self.last_A = A.detach()
        self.last_A = torch.where(mask_ignore, torch.zeros_like(self.last_A), self.last_A).sum() / avg_factor
        self.last_B = B.detach()
        self.last_B = torch.where(mask_ignore, torch.zeros_like(self.last_B), self.last_B).sum() / avg_factor
        self.last_C = C.detach()
        self.last_C = torch.where(mask_ignore, torch.zeros_like(self.last_C), self.last_C).sum() / avg_factor
        self.last_D = D.detach()
        self.last_D = torch.where(mask_ignore, torch.zeros_like(self.last_D), self.last_D).sum() / avg_factor
        self.last_E = E.detach()
        self.last_E = torch.where(mask_ignore, torch.zeros_like(self.last_E), self.last_E).sum() / avg_factor

        self.epoch_num_ = self.epoch_num
        self.iter_cnt += 1
        self.epoch_nums.append(self.epoch_num)

        return self.loss_weight * loss_cls

    def get_logs(self,
                 pred,
                 target,
                 ignore_index=255):
        logs = {}
        pred_detached = pred.detach()
        evidence = self.logit2evidence(pred_detached)
        alpha = (evidence + 1)**2 if self.pow_alpha else evidence + 1
        strength = alpha.sum(dim=1, keepdim=True)
        u = self.num_classes / strength
        prob = alpha / strength
        var = torch.sum(alpha * (strength - alpha) / (strength * strength * (strength + 1)), dim=1)
        max_prob, pred_cls = torch.max(prob.data.clone(), dim=1, keepdim=True)
        gt_cls = target.data.unsqueeze(1).clone()
        mask_ignore = (gt_cls == ignore_index)

        succ = torch.logical_and((pred_cls == gt_cls), ~mask_ignore)
        fail = torch.logical_and(~(pred_cls == gt_cls), ~mask_ignore)
        logs["mean_dir_var"] = torch.where(mask_ignore, torch.zeros_like(var), var).sum() / (~mask_ignore).sum()
        # logs["mean_dir_fake_var"] = torch.where(mask_ignore, torch.zeros_like(var), var).sum() / (~mask_ignore).sum()
        logs["mean_ev_sum"] = evidence.sum(dim=1, keepdim=True).mean()
        logs["mean_fail_ev_sum"] = (evidence.sum(dim=1, keepdim=True) * fail).sum() / (fail.sum() + EPS)
        logs["mean_succ_ev_sum"] = (evidence.sum(dim=1, keepdim=True) * succ).sum() / (succ.sum() + EPS)
        # import ipdb; ipdb.set_trace()
        gt_cls_ = gt_cls.clone()
        gt_cls_[mask_ignore] = 0
        # cls_ev = evidence.gather(1, gt_cls_)
        # for c in range(self.num_classes):
        #     mask_cls = torch.logical_and(gt_cls == c, ~mask_ignore)
        #     if mask_cls.any():
        #         logs[f"mean_target_cls_{c}_ev"] = cls_ev[mask_cls].mean()
        #     else:
        #         logs[f"mean_target_cls_{c}_ev"] = torch.tensor(0., device=alpha.device)
        logs["mean_max_ev"] = evidence.max(dim=1, keepdim=True)[0].mean()

        logs["mean_L0_A"] = self.last_A
        # if self.loss_name.endswith("mse"):
        logs["mean_L1_B"] = self.last_B
        logs["mean_L_kl_C"] = self.last_C
        logs["mean_acc_uncertain_D"] = self.last_D
        logs["mean_inacc_certain_E"] = self.last_E

        logs["avg_max_prob"] = (max_prob * ~mask_ignore).sum() / ((~mask_ignore).sum() + EPS)
        logs["avg_uncertainty"] = (u * ~mask_ignore).sum() / ((~mask_ignore).sum() + EPS)
        # logs["avg_diss"] = (diss(alpha) * ~mask_ignore).sum() / ((~mask_ignore).sum() + EPS)
        logs["acc_seg"] = succ.sum() / (fail.sum() + succ.sum()) * 100

        # max_prob_flat = max_prob.permute(1, 0, 2, 3).flatten(1, -1).squeeze()
        # u_flat = u.permute(1, 0, 2, 3).flatten(1, -1).squeeze()
        # logs["avg_max_prob_u_corr"] = torchmetrics.functional.pearson_corrcoef(max_prob_flat, u_flat)

        # one_hot_gt = torch.zeros_like(pred, dtype=torch.uint8).scatter_(1, gt_cls_, 1)
        # fake_alphas = alpha * (1 - one_hot_gt) + one_hot_gt
        # fake_strength = alpha.sum(dim=1, keepdim=True)
        # fake_var = torch.sum((fake_alphas * (fake_strength - fake_alphas)
        #                       ) / (fake_strength * fake_strength * (fake_strength + 1)), dim=1, keepdim=True)
        # logs["mean_dir_fake_var"] = torch.where(mask_ignore, torch.zeros_like(fake_var), fake_var).sum() / (~mask_ignore).sum()
        # import ipdb; ipdb.set_trace()
        return logs

    # @ property
    # def lam(self):
    #     if self.annealing_method == 'step':
    #         annealing_coef = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(
    #             self.epoch_num / self.annealing_step, dtype=torch.float32))
    #     elif self.annealing_method == 'exp':
    #         # if self.epoch_num + 1 < self.annealing_from:
    #         #     annealing_coef = torch.tensor(0.)
    #         # else:
    #         #     annealing_coef = self.annealing_start * torch.exp(-torch.log(self.annealing_start) / (
    #         #         self.total_epochs - self.annealing_from) * (self.epoch_num + 1 - self.annealing_from))
    #         annealing_coef = self.annealing_start * torch.exp(-torch.log(self.annealing_start) / (self.total_epochs - 1) * self.epoch_num)
    #     elif self.annealing_method == 'zero':
    #         annealing_coef = torch.tensor(0., dtype=torch.float32)
    #     else:
    #         raise NotImplementedError
    #     return annealing_coef
