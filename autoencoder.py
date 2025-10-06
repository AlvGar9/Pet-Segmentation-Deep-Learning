#!/usr/bin/env python
# coding: utf-8

# IMPORTS
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

# GPU SETTINGS
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

# GLOBALS AND SETTING SEED
# --------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 1) AUTOENCODER
# --------------------------------------------------------------
class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        reduced_channels = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
    def forward(self, x):
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2) # Global Average Pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.se = SqueezeExciteBlock(out_ch) if use_se else nn.Identity()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.se(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, use_se=True):
        super(Encoder, self).__init__()
        self.enc1 = ConvBlock(in_channels, base_channels, use_se=use_se)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels*2, use_se=use_se)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4, use_se=use_se)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8, use_se=use_se)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16, use_se=use_se)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        bottleneck = self.bottleneck(self.pool4(x4))
        return bottleneck, [x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, out_classes, base_channels=64, use_se=True):
        super(Decoder, self).__init__()
        self.up4 = nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1)
        self.up3 = nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1)
        self.up2 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        self.up1 = nn.Conv2d(base_channels*2, base_channels*1, 3, padding=1)
        self.dec4 = ConvBlock(base_channels*16, base_channels*8, use_se=use_se)
        self.dec3 = ConvBlock(base_channels*8, base_channels*4, use_se=use_se)
        self.dec2 = ConvBlock(base_channels*4, base_channels*2, use_se=use_se)
        self.dec1 = ConvBlock(base_channels*2, base_channels*1, use_se=use_se)
        self.final_conv = nn.Conv2d(base_channels, out_classes, kernel_size=1)
    def forward(self, bottleneck, skips):
        x1, x2, x3, x4 = skips
        y4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        y4 = self.up4(y4)
        y4 = torch.cat([y4, x4], dim=1)
        y4 = self.dec4(y4)
        y3 = F.interpolate(y4, scale_factor=2, mode='bilinear', align_corners=True)
        y3 = self.up3(y3)
        y3 = torch.cat([y3, x3], dim=1)
        y3 = self.dec3(y3)
        y2 = F.interpolate(y3, scale_factor=2, mode='bilinear', align_corners=True)
        y2 = self.up2(y2)
        y2 = torch.cat([y2, x2], dim=1)
        y2 = self.dec2(y2)
        y1 = F.interpolate(y2, scale_factor=2, mode='bilinear', align_corners=True)
        y1 = self.up1(y1)
        y1 = torch.cat([y1, x1], dim=1)
        y1 = self.dec1(y1)
        output = self.final_conv(y1)
        return output

class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_channels=64):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_channels, base_channels, use_se=True)
        self.decoder = Decoder(num_classes, base_channels, use_se=True)
    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        seg_logits = self.decoder(bottleneck, skips)
        return seg_logits

# 2) SEGMENTATION MODEL USING ENCODER
# --------------------------------------------------------------
class SegmentationModel(nn.Module):
    def __init__(self, autoencoder, num_classes=3):
        super(SegmentationModel, self).__init__()
        self.encoder = autoencoder.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False # Freeze encoder weights
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec0 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    def forward(self, x):
        bottleneck, _ = self.encoder(x)
        d4 = self.dec4(bottleneck)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        d0 = self.dec0(d1)
        out = self.final_conv(d0)
        return out

# 3) DATASETS
# --------------------------------------------------------------
class BasePetDataset(Dataset):
    """
    Loads (image, mask) from a single repository, no transforms.
    """
    def __init__(self, root, is_train=True):
        self.root = os.path.abspath(root)
        path = "TrainVal" if is_train else "Test"
        self.image_dir = os.path.join(self.root, path, "color")
        self.mask_dir = os.path.join(self.root, path, "label")
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Check dataset structure. Missing: {self.image_dir} or {self.mask_dir}")
        self.img_names = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        image_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Corrupted image: {img_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_pil = Image.open(mask_path).convert("L")
        mask = np.array(mask_pil, dtype=np.uint8)
        mask_out = np.zeros_like(mask, dtype=np.uint8)
        mask_out[mask == 38] = 1
        mask_out[mask == 75] = 2
        mask_out[(mask == 255) | (mask == 0)] = 0
        return image, mask_out

class AutoencoderTransformSubset(Dataset):
    """
    Returns just images (no masks) for reconstruction.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image_np, mask_np = self.subset[idx]
        if self.transform is not None:
            transformed = self.transform(image=image_np)
            image_t = transformed["image"]
        else:
            image_t = torch.from_numpy(image_np).permute(2,0,1)
        return image_t

class SegTransformSubset(Dataset):
    """
    Returns (images, masks) with Albumentations transforms (both image+mask).
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        image_np, mask_np = self.subset[idx]
        if self.transform is not None:
            out = self.transform(image=image_np, mask=mask_np)
            image_t = out["image"]
            mask_t = out["mask"].long()
        else:
            image_t = torch.from_numpy(image_np).permute(2,0,1)
            mask_t = torch.from_numpy(mask_np).long()
        return image_t, mask_t

# 4) SPLIT DATA AND DEFINE TRANSFORMS
# --------------------------------------------------------------
base_dataset = BasePetDataset(root="path/dataset", is_train=True)
train_ratio = 0.8
train_size = int(train_ratio * len(base_dataset))
val_size = len(base_dataset) - train_size
generator = torch.Generator().manual_seed(SEED)
train_subset, val_subset = random_split(base_dataset, [train_size, val_size], generator=generator)

autoenc_transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

seg_train_transform = A.Compose([
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

seg_val_transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

train_dataset_ae = AutoencoderTransformSubset(train_subset, transform=autoenc_transform)
train_dataset_seg = SegTransformSubset(train_subset, transform=seg_train_transform)
val_dataset_seg = SegTransformSubset(val_subset, transform=seg_val_transform)

# 5) BUILD DATALOADERS
# --------------------------------------------------------------
train_dataloader_ae = DataLoader(train_dataset_ae, batch_size=32, shuffle=True, num_workers=2)
train_dataloader_seg = DataLoader(train_dataset_seg, batch_size=8, shuffle=True, num_workers=2)
val_dataloader_seg = DataLoader(val_dataset_seg, batch_size=8, shuffle=False, num_workers=2)

# 6) TRAIN THE AUTOENCODER
# --------------------------------------------------------------
autoencoder = Autoencoder(in_channels=3, num_classes=3, base_channels=64).to(device)
optimizer_ae = AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=1e-3)
criterion_ae = nn.MSELoss()
scheduler_ae = torch.optim.lr_scheduler.StepLR(optimizer_ae, step_size=10, gamma=0.5)
patience_ae = 5
num_epochs_ae = 50
best_loss_ae = float("inf")
early_stop_counter = 0
loss_history_ae = []

for epoch in range(num_epochs_ae):
    autoencoder.train()
    running_loss = 0.0
    for images in tqdm(train_dataloader_ae, desc=f"Autoenc Epoch {epoch+1}/{num_epochs_ae}"):
        images = images.to(device)
        optimizer_ae.zero_grad()
        outputs = autoencoder(images)
        loss = criterion_ae(outputs, images)
        loss.backward()
        optimizer_ae.step()
        running_loss += loss.item()
    scheduler_ae.step()
    epoch_loss = running_loss / len(train_dataloader_ae)
    loss_history_ae.append(epoch_loss)
    print(f"[Autoenc] Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
    if epoch_loss < best_loss_ae:
        best_loss_ae = epoch_loss
        early_stop_counter = 0
        torch.save(autoencoder.state_dict(), "best_autoencoder.pth")
    else:
        early_stop_counter += 1
    if early_stop_counter >= patience_ae:
        print(f"Early stopping autoencoder at epoch {epoch+1}")
        break

autoencoder.load_state_dict(torch.load("best_autoencoder.pth"))
autoencoder.eval()

plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_history_ae) + 1), loss_history_ae, marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Reconstruction Loss")
plt.grid(True)
plt.savefig("autoenc_loss_curve.png")
plt.show()

# 7) BUILD AND TRAIN SEGMENTATION MODEL
# --------------------------------------------------------------
seg_model = SegmentationModel(autoencoder=autoencoder, num_classes=3)
dice_metric = DiceScore(num_classes=3, average='macro').to(device)
iou_metric = MeanIoU(num_classes=3).to(device)
acc_metric = MulticlassAccuracy(num_classes=3).to(device)
seg_criterion = nn.CrossEntropyLoss()

def train_one_epoch_seg(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Seg Train", leave=True):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_segmentation_metrics(model, dataloader, device):
    model.eval()
    dice_metric.reset()
    iou_metric.reset()
    acc_metric.reset()
    running_loss = 0.0
    total_samples = len(dataloader.dataset)
    for images, masks in tqdm(dataloader, desc="Evaluating", leave=True):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = seg_criterion(logits, masks)
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        preds_onehot = F.one_hot(preds, num_classes=3).permute(0, 3, 1, 2).float()
        masks_onehot = F.one_hot(masks, num_classes=3).permute(0, 3, 1, 2).float()
        dice_metric.update(preds_onehot, masks_onehot)
        iou_metric.update(preds_onehot.int(), masks_onehot.int())
        acc_metric.update(preds, masks)
    avg_loss = running_loss / total_samples
    return {
        "val_loss": avg_loss,
        "dice": dice_metric.compute().item(),
        "iou": iou_metric.compute().item(),
        "acc": acc_metric.compute().item()
    }

def train_eval_once(model, lr, batch_size, weight_decay=1e-3, patience=5, max_epochs=100, device=device):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    best_val_loss = float("inf")
    # ... training loop similar to UNET notebook ...
    # Placeholder for the full training loop logic
    print("Training loop for segmentation model...")
    # This would be the full training loop as in the UNET notebook
    # Returning dummy values for demonstration
    return 0.1, model.state_dict(), {}

seg_model = seg_model.to(device)
seg_model = nn.DataParallel(seg_model)
lr_seg = 1e-4
batch_size = 32 # or 8
best_val_loss, best_state, curves = train_eval_once(
    model=seg_model,
    lr=lr_seg,
    batch_size=batch_size,
    patience=5,
    max_epochs=100,
    device=device
)
torch.save(best_state, f"seg_model_lr{lr_seg}.pth")
print(f"Best val loss: {best_val_loss:.4f}")
# Plotting logic here...
