import matplotlib.pyplot as plt
import torch
EPS = 1e-5
logits = torch.arange(-100., 100., 0.01)
# logits.requires_grad = True


def exp(logits):
    mask_pos = (logits >= 0)
    mask_neg = ~mask_pos
    ans = torch.zeros_like(logits)
    ans[mask_pos] = 1 / (EPS + torch.exp(-logits[mask_pos]))
    ans[mask_neg] = torch.exp(logits[mask_neg])
    return ans


# ans = exp(logits)
# import ipdb; ipdb.set_trace()

# output = 1. / (torch.exp(-1 * logits) + EPS)
fig = plt.figure()

# plt.plot(logits, 1. / (torch.exp(-.1 * logits) + EPS), label="inv_.1", color='b')
plt.plot(logits, torch.exp(logits), label="regular", color='g')
plt.plot(logits, exp(logits), label="inv", color='r')
plt.plot(logits, (torch.abs(logits) + 1)**1.5, label="inv", color='k')
plt.axis([None, None, -1, 500])
plt.savefig("exp_eps_jj.png", dpi=300)
# import ipdb; ipdb.set_trace()
