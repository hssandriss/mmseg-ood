import torch
annealing_start = torch.tensor(0.001, dtype=torch.float32)
total_epochs = 70
for epoch_num in range(total_epochs):
    # if epoch_num + 1 < annealing_from:
    #     annealing_coef = torch.tensor(0.)
    # else:
    annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / (total_epochs - 1) * epoch_num)
    # annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / (
    #     total_epochs - ) * (epoch_num + 1 - annealing_from))
    print(f"epoch {epoch_num+1}- {annealing_coef.item()}")
