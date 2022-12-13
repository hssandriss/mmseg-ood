import torch 

for epoch in [10]:
    path = f"../mmrazor/work_dirs/deeplabv3_r50-d8_720x720_70e_cityscapes_cwd_20221011180853_lr=0.01_bs=8_cwd_weight=0.1/epoch_{epoch}.pth"
    ckpt = torch.load(path)
    architecture_keys = [k for k in ckpt['state_dict'].keys() if k.startswith('architecture')]
    for k in architecture_keys:
        ckpt['state_dict'][k.replace('architecture.model.', '')] = ckpt['state_dict'][k]
    torch.save(ckpt, path)
    print(f"[*] Epoch {epoch} done!")