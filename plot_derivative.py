import matplotlib.pyplot as plt
import torch
EPS = 1e-5
logits = torch.arange(-20., 20., 0.01)

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
    return  2*torch.nn.functional.softplus(logits)*softplus_derivative(logits) + 2*softplus_derivative(logits)

fig = plt.figure()
f_elu = elu_derivative(logits)
f_exp = exp(logits)
f_softplus = softplus_derivative(logits)
f_softplus_bended = softplus_poly_derivative(logits)

plt.plot(logits, f_elu, label='elu',linewidth=1)
plt.plot(logits, f_exp, label='exp', linewidth=1)
plt.plot(logits, f_softplus, label='softplus', linewidth=1)
plt.plot(logits, f_softplus_bended, label='softplus polynomial', linewidth=1)
plt.axvline(4.5, linestyle='dashed', color='k', linewidth=.5)
plt.text(3.8, 10, 'a=4.5',rotation='vertical', fontsize=6, )
plt.axvline(5,linestyle='dashed', color='k', linewidth=.5)
plt.text(5.1, 10, 'b=5',rotation='vertical', fontsize=6)
plt.axis([None, None, -1, 100])
plt.title("Different activations derivatives")
plt.legend()
plt.savefig("derivative.png", dpi=300)