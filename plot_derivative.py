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


def elu_derivative(logits):
    mask_pos = (logits > 0)
    mask_neg = ~mask_pos
    ans = torch.zeros_like(logits)
    ans[mask_pos] = 1
    ans[mask_neg] = exp(logits[mask_neg])
    return ans


def softplus_derivative(logits):
    return torch.sigmoid(logits)


def softplus_poly_derivative(logits):
    return 2 * torch.nn.functional.softplus(logits) * softplus_derivative(logits) + 2 * softplus_derivative(logits)


def elu_poly_derivative(logits):
    return 2 * (torch.nn.functional.elu(logits) + 1) * elu_derivative(logits) + 2 * elu_derivative(logits)


# def softplus_sq_derivative(logits):
#     return  2*torch.nn.functional.softplus(logits)*softplus_derivative(logits)

# def elu_sq_derivative(logits):
#     return  2*(torch.nn.functional.elu(logits)+1)*elu_derivative(logits)

fig = plt.figure()
f_exp = exp(logits)
f_elu = elu_derivative(logits)
f_softplus = softplus_derivative(logits)
f_softplus_poly = softplus_poly_derivative(logits)
f_elu_poly = elu_poly_derivative(logits)


# f_elu_sq_bended = elu_sq_derivative(logits)
# f_softplus_sq_bended = elu_sq_derivative(logits)


plt.plot(logits, f_exp, label='exp', linewidth=1)
plt.plot(logits, f_elu, label='elu', linewidth=1)
plt.plot(logits, f_softplus, label='softplus', linewidth=1)
plt.plot(logits, f_softplus_poly, label='softplus²+2.softplus', linewidth=1)
plt.plot(logits, f_elu_poly, label='elu²+2.elu', linewidth=1)
# plt.axvline(-5, linestyle='dashed', color='k', linewidth=.5)
# plt.axvline(5,linestyle='dashed', color='k', linewidth=.5)
plt.axis([None, None, -1, 15])
plt.title("Different activations derivatives")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=5)
plt.savefig("derivative_.png", dpi=300, bbox_inches='tight')
