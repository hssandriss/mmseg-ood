"""
Implementing cc fusion for sl
https://github.com/vs-uulm/subjective-logic-java/blob/master/src/no/uio/subjective_logic/opinion/SubjectiveOpinion.java
Another paper:
Expert Opinion Fusion Framework Using Subjective Logic for Fault Diagnosis
"""


import torch
import itertools
EPS = 1e-9
nsamples = 2
ev = torch.relu(torch.randn(size=(nsamples, 19, 720, 1280)))
# Base Rate should be changed to 1/(K-1)
a = 1 / (ev.size(1) - 1)
W = ev.size(1)
s = (ev + 1).sum(1, keepdim=True)
bel = ev / s
u = W / s


bel_cons_x = bel.min(dim=0, keepdim=True)[0]
bel_cons = bel_cons_x.sum(1, keepdim=True)
bel_res_x = torch.relu(bel - bel_cons_x)


bel_comp_x = (bel_res_x * u.prod(0, keepdim=True) / (u + EPS)).sum(0, keepdim=True)  # first part of sum [1, 19, 720, 1280]
u_comp = torch.zeros_like(u)
COMB_LIST = list(itertools.combinations_with_replacement(range(ev.size(1)), r=ev.size(0)))
for comb in COMB_LIST:
    if len(set(comb)) == 1:
        # Intersection at x
        bel_comp_x[0, comb[0], :, :] += a**ev.size(0) * bel_res_x[range(ev.size(0)), comb, :, :].prod(0)
        # Union at x and non null intersection
        bel_comp_x[0, comb[0], :, :] += (1 - a**ev.size(0)) * bel_res_x[range(ev.size(0)), comb, :, :].prod(0)
    else:
        # Add this to ucomp
        # Union at x and null intersection
        prod = bel_res_x[range(ev.size(0)), comb, :, :].prod(0)
        for c in set(comb):
            bel_comp_x[0, c, :, :] += prod

bel_comp = bel_comp_x.sum(1, keepdim=True)
u_pre = u.prod(0, keepdim=True)

test_sum = u_pre + bel_comp + bel_cons
import ipdb; ipdb.set_trace()

nu = (1 - bel_cons - u_pre) / (bel_comp)

comb_bel = bel_cons_x + nu * bel_comp_x
comb_u = u_pre + nu * bel_comp
import ipdb; ipdb.set_trace()
comb_test_sum = comb_u + comb_bel.sum(1, keepdim=True)
