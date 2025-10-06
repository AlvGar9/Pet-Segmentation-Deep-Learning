#!/usr/bin/env python
# coding: utf-8

# 1) IMPORTS
# --------------------------------------------------------------
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import MulticlassAccuracy
from transformers import CLIPModel

# 2) SETUP: GPU SETTINGS
# --------------------------------------------------------------
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Version of CUDA:", torch.version.cuda)
    print("Number of devices:", torch.cuda.device_count())
    print("Current device index:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)
print()

# 3) GLOBALS SETTING AND SEED
# --------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 4) NETWORK ARCHITECTURE
# --------------------------------------------------------------
class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcite, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class HFCLIPVisionEncoder(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14"):
        super(HFCLIPVisionEncoder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.hidden_dim = self.clip_model.vision_model.config.hidden_size
    def forward(self, x):
        vision_outputs = self.clip_model.vision_model(pixel_values=x)
        last_hidden = vision_outputs.last_hidden_state
        patch_tokens = last_hidden[:, 1:, :]
        b, n, c = patch_tokens.shape
        feat_dim = int(n ** 0.5)
        patch_tokens_2d = patch_tokens.permute(0, 2, 1).contiguous()
        feature_map = patch_tokens_2d.view(b, c, feat_dim, feat_dim)
        return feature_map, None

class SegmentationModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", num_classes=3):
        super(SegmentationModel, self).__init__()
        self.encoder = HFCLIPVisionEncoder(clip_model_name=clip_model_name)
        in_channels = self.encoder.hidden_dim
        self.dec4 = self._make_decoder(in_channels, 512)
        self.dec3 = self._make_decoder(512, 256)
        self.dec2 = self._make_decoder(256, 128)
        self.dec1 = self._make_decoder(128, 64)
        self.final_upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    def _make_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            SqueezeExcite(out_channels)
        )
    def forward(self, x):
        bottleneck, _ = self.encoder(x)
        d4 = self.dec4(bottleneck)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        return out

# 5) DATASET CLASS & TRANSFORMS
# --------------------------------------------------------------
# (Using BasePetDataset and TransformSubset from A.2)
# ...
# The `BasePetDataset` and `TransformSubset` classes are the same as in the autoencoder notebook.
class BasePetDataset(Dataset):
    def __init__(self, root, is_train=True):
        self.root = os.path.abspath(root)
        path = "TrainVal" if is_train else "Test"
        self.image_dir = os.path.join(self.root, path, "color")
        self.mask_dir = os.path.join(self.root, path, "label")
        self.img_names = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        image_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_pil = Image.open(mask_path).convert("L")
        mask = np.array(mask_pil, dtype=np.uint8)
        mask_out = np.zeros_like(mask, dtype=np.uint8)
        mask_out[mask == 38] = 1 # Cat
        mask_out[mask == 75] = 2 # Dog
        mask_out[(mask == 255) | (mask == 0)] = 0 # Background
        return image, mask_out

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image_np, mask_np = self.subset[idx]
        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_transformed = transformed["image"]
            mask_transformed = transformed["mask"].long()
        else:
            image_transformed = torch.from_numpy(image_np).permute(2, 0, 1).float()
            mask_transformed = torch.from_numpy(mask_np).long()
        return image_transformed, mask_transformed

base_dataset = BasePetDataset(root="path/dataset", is_train=True)
train_ratio = 0.8
train_size = int(train_ratio * len(base_dataset))
val_size = len(base_dataset) - train_size
generator = torch.Generator().manual_seed(SEED)
train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

train_transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

train_dataset = TransformSubset(subset=train_subset, transform=train_transform)
val_dataset = TransformSubset(subset=val_subset, transform=val_transform)

# 7) METRICS, FUNCTIONS, AND TRAINING
# --------------------------------------------------------------
# (Using train_one_epoch and evaluate_segmentation_metrics from A.2)
# ...
# The functions `train_one_epoch` and `evaluate_segmentation_metrics` are the same.
criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer, criterion, device):
    # ... (same as before)
def evaluate_segmentation_metrics(model, dataloader, device):
    # ... (same as before)

def train_eval_once(lr, batch_size, weight_decay=1e-3, patience=5, max_epochs=100, device=device):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model = SegmentationModel(num_classes=3).to(device)
    model = nn.DataParallel(model)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    
    # ... (Full training loop as in UNET notebook)
    # Placeholder for the full training loop logic
    print("Training loop for CLIP-based model...")
    # Returning dummy values for demonstration
    return 0.1, model.state_dict(), {}

learning_rate = 1e-4 # Best from search
batch_size = 32    # Best from search
val_loss, model_state, curves = train_eval_once(
    lr=learning_rate,
    batch_size=batch_size,
    patience=5,
    max_epochs=100,
    device=device
)
# Plotting logic here...
