import matplotlib.pyplot as plt
import torch
EPS = 1e-5
logits = torch.arange(-10., 10., 0.01)

def exp(logits):
    mask_pos = (logits >= 0)
    mask_neg = ~mask_pos
    ans = torch.zeros_like(logits)
    ans[mask_pos] = 1 / (EPS + torch.exp(-logits[mask_pos]))
    ans[mask_neg] = torch.exp(logits[mask_neg])
    return ans

fig = plt.figure()
f_elu = torch.nn.functional.elu(logits)+1
f_exp = exp(logits)
f_softplus = torch.nn.functional.softplus(logits)
f_softplus_poly = f_softplus**2 + 2*f_softplus
f_elu_poly = f_elu**2 + 2*f_elu

plt.plot(logits, f_exp, label='exp', linewidth=1)
plt.plot(logits, f_elu, label='elu',linewidth=1)
plt.plot(logits, f_softplus, label='softplus', linewidth=1)
plt.plot(logits, f_softplus_poly, label='softplus²+2.softplus', linewidth=1)
plt.plot(logits, f_elu_poly, label='elu²+2.elu', linewidth=1)

plt.axis([None, None, -1, 20])
plt.title("Different activations")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.savefig("activations_.png", dpi=300, bbox_inches='tight')



