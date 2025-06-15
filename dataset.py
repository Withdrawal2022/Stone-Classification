# dataset.py
from pathlib import Path
import json, collections, pandas as pd
import torch, random
from torch.utils.data import Dataset

__all__ = ["StoneDatasetCached", "fine2idx", "fine2coarse"]

# ---------------- 可调参数 ----------------
CACHE_DIR       = "cache_384"   # 离线图片保存文件夹
RESIZE_LONGSIDE = 384           # 离线统一长(短)边
CROP_TRAIN      = 300           # 训练随机裁剪尺寸
CROP_EVAL       = 384           # 验证/测试裁剪尺寸
# -----------------------------------------

########################################################################
# 细→粗 / 细→idx  映射生成
########################################################################
def _build_mapping(csv_files, min_samples=30, cache="fine_mapping.json"):
    cache = Path(cache)
    if cache.exists():
        m = json.load(cache.open())
        return {k:int(v) for k,v in m["fine2coarse"].items()}, \
               {k:int(v) for k,v in m["fine2idx"].items()}

    ctr, fine2coarse_tmp = collections.Counter(), {}
    for csv in csv_files:
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            fine = str(row["fine"]).strip()
            ctr[fine] += 1
            fine2coarse_tmp[fine] = int(row["coarse"])

    fine2idx, fine2coarse, idx = {}, {}, 0
    for fine, cnt in ctr.items():
        coarse = fine2coarse_tmp[fine]
        if cnt < min_samples:
            fine = f"other_{coarse}"
        if fine not in fine2idx:
            fine2idx[fine] = idx; idx += 1
        fine2coarse[fine] = coarse

    json.dump({"fine2coarse":fine2coarse,"fine2idx":fine2idx},
              cache.open("w"), ensure_ascii=False, indent=2)
    return fine2coarse, fine2idx


########################################################################
# Dataset  — 从 .pt 缓存读取，极快
########################################################################
# 新增两个小辅助
def cutout(t, size=64):
    _, H, W = t.shape
    y = random.randint(0, H-size)
    x = random.randint(0, W-size)
    t[:, y:y+size, x:x+size] = t.mean()
    return t

def blur_score(t):                       # tensor 0-1
    lap = torch.nn.functional.avg_pool2d(t.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0] - t
    return lap.abs().mean().item()       # <0.05 视为糊

class StoneDatasetCached(Dataset):
    """
    离线缓存路径:
        dataset/cache_384/train_val/train/xxx.jpg.pt
        (= 3×384×384 uint8)、val/、test/test_images/
    """
    def __init__(self, root, split="train", transforms=None, min_samples=30,
                 cache_json="fine_mapping.json"):
        self.root  = Path(root)
        self.cache = self.root / CACHE_DIR
        self.split = split.lower()     # train | val | test
        self.transforms = transforms      # 保存

        tv_dir = self.root / "train_val"
        train_csv = tv_dir/"train_labels.csv"
        val_csv   = tv_dir/"val_labels.csv"
        self.fine2coarse, self.fine2idx = _build_mapping(
            [train_csv, val_csv], min_samples=min_samples, cache=cache_json
        )

        if self.split == "train":
            df = pd.read_csv(train_csv, header=0,
                                    names=["fname","coarse","fine"])
            self.img_dir = "train_val/train"
        elif self.split == "val":
            df = pd.read_csv(val_csv, header=0,
                                    names=["fname","coarse","fine"])
            self.img_dir = "train_val/val"
        else:
            test_dir = self.root/"test"
            df = pd.read_csv(test_dir / "test_ids.csv", header=0, names=["fname"])
            self.img_dir = "test/test_images"

        self.samples = []
        for _, row in df.iterrows():
            fname = str(row["fname"]).strip()
            if self.split == "test":
                self.samples.append((fname, None, None)); continue
            coarse = int(row["coarse"])
            fine   = str(row["fine"]).strip()
            if fine not in self.fine2idx:
                fine = f"other_{coarse}"
            self.samples.append((fname, self.fine2idx[fine], self.fine2coarse[fine]))

        # 记录 quality_weight
        self.qweights = []
        for fname, _, _ in self.samples:
            tensor = torch.load(self.cache/self.img_dir/f"{fname}.pt", weights_only = True)
            q = blur_score(tensor.float()/255.)
            self.qweights.append(0.2 if q < 0.05 else 1.0)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, fine_id, coarse_id = self.samples[idx]
        tensor = torch.load(self.cache / self.img_dir / f"{fname}.pt", weights_only = True)  # uint8 C×H×W

        # --- 轻量级在线增强 ---
        if self.split == "train":
            y = random.randint(0, RESIZE_LONGSIDE - CROP_TRAIN)
            x = random.randint(0, RESIZE_LONGSIDE - CROP_TRAIN)
            tensor = tensor[:, y:y+CROP_TRAIN, x:x+CROP_TRAIN]
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
        else:
            m = (RESIZE_LONGSIDE - CROP_EVAL)//2
            tensor = tensor[:, m:m+CROP_EVAL, m:m+CROP_EVAL]
            
        # --- ② 转 float & 0‑1 ---
        tensor = tensor.float() / 255.0

        if self.split == "train":
            # # ① 额外颜色抖动 / Cutout
            # if random.random() < .5:
            #     tensor = tensor * (1 + (torch.rand(3,1,1)*0.1 - 0.05))
            #     tensor = tensor.clamp(0,1)
            if random.random() < .2:
                tensor = cutout(tensor)

        # --- ③ 额外 Tensor 级增强（可选） ---
        if self.transforms is not None:
            tensor = self.transforms(tensor)

        # --- ④ Normalize ---
        MEAN = torch.tensor([0.4602, 0.4554, 0.4505]).view(3,1,1)
        STD  = torch.tensor([0.2840, 0.2832, 0.2888]).view(3,1,1)
        
        tensor = (tensor - MEAN) / STD
        

        if self.split == "test":
            return tensor, fname
        return (tensor, fine_id, coarse_id, self.qweights[idx]) if self.split=="train" \
               else (tensor, fname) if self.split=="test" \
               else (tensor, fine_id, coarse_id)

# # 暴露映射
# fine2idx, fine2coarse = None, None



# from PIL import Image, UnidentifiedImageError
# import os, json, collections
# from pathlib import Path
# import pandas as pd
# from torch.utils.data import Dataset

# __all__ = ["StoneDataset", "fine2idx", "fine2coarse"]

# ########################################################################
# # 1. 自动生成 细→粗 / 细→idx 映射（只在首次运行时扫描一次）
# ########################################################################
# def _build_mapping(csv_files, min_samples=30, cache="fine_mapping.json"):
#     if Path(cache).exists():
#         m = json.load(open(cache, "r", encoding="utf-8"))
#         return {k: int(v) for k, v in m["fine2coarse"].items()}, \
#                {k: int(v) for k, v in m["fine2idx"].items()}

#     # 统计细粒度出现次数
#     counter = collections.Counter()
#     fine_coarse_tmp = {}
#     for csv in csv_files:
#         df = pd.read_csv(csv, header=0, names=["fname", "coarse", "fine"])
#         for _, row in df.iterrows():
#             fine = str(row["fine"]).strip()
#             counter[fine] += 1
#             fine_coarse_tmp[fine] = int(row["coarse"])

#     fine2idx, fine2coarse = {}, {}
#     idx = 0
#     for fine, cnt in counter.items():
#         coarse = fine_coarse_tmp[fine]
#         if cnt < min_samples:
#             fine = f"other_{coarse}"
#         if fine not in fine2idx:
#             fine2idx[fine] = idx; idx += 1
#         fine2coarse[fine] = coarse

#     json.dump({"fine2coarse": fine2coarse, "fine2idx": fine2idx},
#               open(cache, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
#     return fine2coarse, fine2idx

# class StoneDataset(Dataset):
#     """
#     root/
#         train_val/train_labels.csv
#         train_val/val_labels.csv
#         train_val/train/*.jpg
#         train_val/val/*.jpg
#         test/test_ids.csv
#         test/test_images/*.jpg
#     """
#     def __init__(self, root, split="train", transforms=None,
#                  cache_json="fine_mapping.json", min_samples=30):
#         self.root = Path(root)
#         self.split = split.lower()  # "train" | "val" | "test"
#         self.transforms = transforms

#         # --- 映射 ---
#         tv_dir = self.root / "train_val"
#         train_csv = tv_dir / "train_labels.csv"
#         val_csv   = tv_dir / "val_labels.csv"
#         self.fine2coarse, self.fine2idx = _build_mapping(
#             [train_csv, val_csv], min_samples=min_samples, cache=cache_json
#         )

#         # --- 读 CSV ---
#         if self.split == "train":
#             self._df = pd.read_csv(train_csv, header=0,
#                                    names=["fname","coarse","fine"])
#             self.img_dir = tv_dir / "train"
#         elif self.split == "val":
#             self._df = pd.read_csv(val_csv, header=0,
#                                    names=["fname","coarse","fine"])
#             self.img_dir = tv_dir / "val"
#         else:  # test
#             test_dir = self.root / "test"
#             ids = pd.read_csv(test_dir / "test_ids.csv", header=0, names=["fname"])
#             self._df = ids
#             self.img_dir = test_dir / "test_images"

#         # 预生成索引表
#         self.samples = []
#         for _, row in self._df.iterrows():
#             fname = str(row["fname"]).strip()
#             if self.split == "test":
#                 self.samples.append((self.img_dir / fname, None, None))
#                 continue
#             coarse_id = int(row["coarse"])
#             fine_name = str(row["fine"]).strip()
#             if fine_name not in self.fine2idx:
#                 fine_name = f"other_{coarse_id}"
#             fine_id   = self.fine2idx[fine_name]
#             coarse_id = self.fine2coarse[fine_name]
#             self.samples.append((self.img_dir / fname, fine_id, coarse_id))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, fine_id, coarse_id = self.samples[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transforms:
#             image = self.transforms(image)
#         if self.split == "test":
#             return image, img_path.name   # 测试集仅返回文件名以便提交
#         return image, fine_id, coarse_id

# # 在 import * 时暴露映射
# fine2coarse, fine2idx = None, None



##### OLD Versions #####
# class StoneDataset(Dataset):
#     def __init__(self, root, split="train", transforms=None):
#         """
#         root: 数据集根目录
#         split: 'train', 'val', 或 'test'
#         transforms: 图像预处理变换
#         """
#         # Train size: 102213, Val size: 15000, Test size: 15000
#         self.root = root
#         self.split = split
#         self.transforms = transforms
#         self.samples = []
#         self.labels = []

#         # 根据 split 加载对应的数据
#         if split in ["train", "val"]:
#             # 加载训练集或验证集
#             csv_path = os.path.join(root, f"{split}_labels.csv")
#             if not os.path.exists(csv_path):
#                 raise FileNotFoundError(f"{csv_path} not found.")
            
#             # 读取 CSV 文件
#             df = pd.read_csv(csv_path)
#             for _, row in df.iterrows():
#                 img_path = os.path.join(root, split, row["id"])
#                 self.samples.append(img_path)
#                 self.labels.append(int(row["label"]))  # label 已为 0, 1, 2
#         elif split == "test":
#             # 加载测试集（无标签，仅图像路径）
#             test_ids_path = os.path.join(root, "test_ids.csv")
#             if not os.path.exists(test_ids_path):
#                 raise FileNotFoundError(f"{test_ids_path} not found.")
            
#             # 读取测试集 ID
#             df = pd.read_csv(test_ids_path)
#             for _, row in df.iterrows():
#                 img_path = os.path.join(root, "test_images", row["id"])
#                 self.samples.append(img_path)
#                 self.labels.append(None)  # 测试集无标签，占位符
#         else:
#             raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")

#     def __getitem__(self, index):
#         img_path = self.samples[index]
#         label = self.labels[index]

#         # 加载图像
#         image = Image.open(img_path).convert("RGB")

#         if self.transforms is not None:
#             image = self.transforms(image)

#         # 对于测试集，label 为 None，仅返回图像
#         if self.split == "test":
#             return image, img_path  # 返回图像和路径以便生成提交文件
#         return image, label  # 返回图像和标签

#     def __len__(self):
#         return len(self.samples)


# if __name__ == "__main__":

#     from torchvision import transforms

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     # 加载数据集
#     dataset_train = StoneDataset(root="./dataset/train_val", split="train", transforms=transform)
#     dataset_val = StoneDataset(root="./dataset/train_val", split="val", transforms=transform)
#     dataset_test = StoneDataset(root="./dataset/test", split="test", transforms=transform)   # for Kaggle test only

#     print(f"Train size: {len(dataset_train)}")
#     print(f"Val size: {len(dataset_val)}")
#     print(f"Test size: {len(dataset_test)}")

#     # 测试加载
#     img, label = dataset_train[0]
#     print(f"Sample image shape: {img.shape}, Label: {label}")