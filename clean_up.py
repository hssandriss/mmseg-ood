tokeep = [
    "deeplabv3_r50-d8_512x512_70e_street_hazards_20220821215145_lr=0.01_bs=4",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_20221023174825_lr=0.01_bs=4_ema_0.0002_warmup_10e",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20220913121145_lr=0.01_bs=3_mll_elu_pow",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20221117131214_lr=0.01_bs=4_mll_elu_pow",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20221115174209_lr=0.01_bs=4_mll_bis_elu_pow",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20221113232503_lr=0.01_bs=4_mll_reg_0.001_elu_pow",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20221113232756_lr=0.01_bs=4_mll_lvar_0.1_elu_pow",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20221113121037_lr=0.01_bs=4_mll_elu_pow_kl_step_a",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_bll_20221124194227_lr=0.001_Adam_bs=4x10_mll_elu_pow_4xsylvester_cnt_transforms=8_1024D_1xfc_layer_no_bias",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_bll_20221124195722_lr=0.001_bs=4x10_mll_elu_pow_2xnaf_flow_default_1024D_1xfc_layer_no_bias",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_bll_20221125133419_lr=0.001_Adam_bs=4x10_mll_elu_pow_full_normal_1024D_1xfc_layer_no_bias",
    "deeplabv3_r50-d8_512x512_70e_street_hazards_edl_20221130190624_lr=0.001_bs=4_mll_elu_pow_ablation_from_ce_30e",

    "deeplabv3_r50-d8_720x720_70e_cityscapes_20221010232119_lr=0.01_bs=8",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_20221010232119_lr=0.01_bs=8_ema_momentum=0.0002_warmup_10e",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20220930205735_lr=0.01_bs=8_mll_elu_pow",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221001195333_lr=0.01_bs=8_mll_bis_elu_pow",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221002100457_lr=0.01_bs=8_mll_new_reg_0.001_elu_pow",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221114230542_lr=0.01_bs=8_mll_lvar_0.1_elu_pow",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221003192529_lr=0.01_bs=8_mll_elu_pow_kl_step_a",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_bll_20221113182902_lr=0.001_bs=8x5_mll_elu_pow_full_normal_1024D_1xfc_layer_no_bias",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_bll_20221115090400_lr=0.001_Adam_bs=8x4_mll_elu_pow_2xnaf_default_1024D_1xfc_layer_no_bias",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_bll_20221116124111_lr=0.001_Adam_bs=4x5_mll_elu_pow_4xsylvester_cnt_transforms=8_1024D_1xfc_layer_no_bias",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221130192019_lr=0.001_bs=4_mll_elu_pow_ablation_from_ce_30e",

    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221130015711_lr=0.01_bs=8_mll_exp_2",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20220921191256_lr=0.01_bs=8_mll_softplus_pow",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20220923152020_lr=0.01_bs=8_mll_softplus^2",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221112161551_lr=0.01_bs=8_mll_elu²",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20220925195153_lr=0.01_bs=8_mll_softplus",
    "deeplabv3_r50-d8_720x720_70e_cityscapes_edl_20221129142829_lr=0.01_bs=8_mll_elu",

    "segmenter_vit-s_mask_720x720_120e_cityscapes_20221001204529_lr=0.001_bs=8",
    "segmenter_vit-s_mask_720x720_120e_cityscapes_20221102164136_lr=0.001_bs=8_ema_momentum=0.0002_warmup_10e",
    "segmenter_vit-s_mask_720x720_120e_cityscapes_edl_20221007095046_lr=0.001_bs=8_mll_exp_2",
    "segmenter_vit-s_mask_720x720_120e_cityscapes_edl_20221008193027_lr=0.001_bs=8_mll_bis_exp_2",
    "segmenter_vit-s_mask_720x720_120e_cityscapes_edl_20221008193049_lr=0.001_bs=8_mll_var_0.1_exp_2",
    "segmenter_vit-s_mask_720x720_120e_cityscapes_edl_20221008193023_lr=0.001_bs=8_mll_reg_0.001_exp_2",
    "segmenter_vit-s_mask_720x720_120e_cityscapes_edl_20221114231140_lr=0.001_bs=8_mll_kl_step_a_exp_2",

    "segmenter_vit-s_mask_512x512_120e_streethazards_20220912221953_lr=0.001_bs=4",
    "segmenter_vit-s_mask_512x512_120e_streethazards_20221102163844_lr=0.001_bs=4_ema_momentum=0.0002_warmup_10e",
    "segmenter_vit-s_mask_512x512_120e_streethazards_edl_20221011184419_lr=0.001_bs=4_mll_exp_2",
    "segmenter_vit-s_mask_512x512_120e_streethazards_edl_20221011184655_lr=0.001_bs=4_mll_bis_exp_2",
    "segmenter_vit-s_mask_512x512_120e_streethazards_edl_20221018235425_lr=0.001_bs=4_mll_new_var_0.1_exp_2",
    "segmenter_vit-s_mask_512x512_120e_streethazards_edl_20221106114331_lr=0.001_bs=8_mll_reg_0.001_exp_2",
    "segmenter_vit-s_mask_512x512_120e_streethazards_edl_20221114231229_lr=0.001_bs=8_mll_kl_step_a_exp_2",
]

import os
import shutil
alldirs = os.listdir('work_dirs')
deleted = []
kept = len(tokeep)
for item in alldirs:
    if item in tokeep:
        kept-=1
        tokeep.remove(item)
        print("keeping: ", item)
    else:
        shutil.rmtree(os.path.join('work_dirs', item))
        print("deleting: ", item)
import ipdb; ipdb.set_trace()
