#!/usr/bin/env python
from pathlib import Path; import os, tqdm, multiprocessing as mp
import numpy as np, torch
from PIL import Image

ROOT   = Path("./dataset")          # 数据根
CACHE  = ROOT/"cache_384"
SRC_DIRS = [
    "train_val/train", "train_val/val", "test/test_images"
]


def smart_crop(pil_img, long_side=384):
    import numpy as np
    img = np.asarray(pil_img.convert("L"))      # H×W  uint8
    H, W = img.shape

    # ① 裁掉多余像素，使 H,W 能被 8 整除
    H8, W8 = (H // 8) * 8, (W // 8) * 8
    img8 = img[:H8, :W8]                        # 丢最底 & 最右不足 8px 的边

    # ② 16×16 Patch 方差图（= 8×8 pooling 两次）
    var = (img8.astype(np.float32)
           .reshape(H8 // 8, 8, W8 // 8, 8)
           .var(axis=(1, 3)))                   # shape (H8/8, W8/8)

    mask = var > var.mean() + 0 * var.std()
    ys, xs = np.where(mask)
    if ys.size < 4:            # ROI 太小 ⇒ 用原图
        return pil_img

    y1, y2 = ys.min() * 8, ys.max() * 8 + 8
    x1, x2 = xs.min() * 8, xs.max() * 8 + 8

    roi = pil_img.crop((x1, y1, x2, y2))
    return roi


def process_one(p):
    in_path, out_path = p
    if out_path.exists(): return
    try:
        img = Image.open(in_path).convert("RGB")
    except Exception as e:
        print("❌", in_path, e); return
    # 等比例缩放: 长边=384, 再居中填充
    # w,h = img.size
    # scale = 384 / max(w,h)
    # img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR) # 等比例缩放
    img = smart_crop(img)
    img.thumbnail((384,384), Image.BILINEAR)          # 等比例缩放
    new = Image.new("RGB", (384,384), (128,128,128)) # 居中
    offset = ((384-img.width)//2, (384-img.height)//2)
    new.paste(img, offset)
    arr = np.array(new)             # HWC uint8
    tensor = torch.from_numpy(arr).permute(2,0,1)   # C,H,W
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_path, _use_new_zipfile_serialization=False)

jobs = []
for sub in SRC_DIRS:
    for jpg in (ROOT/sub).glob("*.jpg"):
        out = CACHE/sub/f"{jpg.name}.pt"
        jobs.append((jpg, out))

print("Total imgs:", len(jobs))
with mp.Pool(mp.cpu_count()) as pool:
    list(tqdm.tqdm(pool.imap_unordered(process_one, jobs), total=len(jobs)))
print("✅  Finished caching.")
