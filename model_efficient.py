import timm, torch.nn as nn, torch
from safetensors.torch import load_file

class EfficientV2S_MTL(nn.Module):
    """
    EfficientNet‑V2‑S backbone + 双输出头（细/粗）
    """
    def __init__(self, n_fine, n_coarse=3, pretrained_ckpt="tf_efficientnetv2_s_in21k.safetensors", drop_rate=0.3):
        super().__init__()
        # 1. backbone 去掉分类头，输出全局特征
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k",
            pretrained=False,
            num_classes=0,          # 删除原 head
            global_pool="avg"       # 得到 [B, 1280]
        )
        self.backbone.load_state_dict(load_file(pretrained_ckpt), strict=False)  # 修改路径
        out_dim = self.backbone.num_features

        # 2. optional dropout
        # self.dropout = nn.Dropout(drop_rate)

        # 3. 两个任务头
        self.head_fine   = nn.Linear(out_dim, n_fine)
        self.head_coarse = nn.Linear(out_dim, n_coarse)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)          # [B, 1280]
        # feat = self.dropout(feat)
        if return_feat:
            return feat
        return self.head_fine(feat), self.head_coarse(feat)
