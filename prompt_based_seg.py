#################################### IMPORTS ########################################
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

#################################### GPU SETTINGS ########################################
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

#################################### GLOBALS AND SETTING SEED ########################################
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#################################### DATASET CLASS ########################################
class PointBasedPetDataset(Dataset):
    def __init__(self, root="path/dataset", is_train=True, transform=None, points_per_image=5):
        self.transform = transform
        self.classes = ['background', 'cat', 'dog', 'border']
        self.root = os.path.abspath(root)
        self.points_per_image = points_per_image
        path = "TrainVal" if is_train else "Test"
        self.image_dir = os.path.join(self.root, path, "color")
        self.mask_dir = os.path.join(self.root, path, "label")
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Check dataset structure. Missing: {self.image_dir} or {self.mask_dir}")
        self.img_names = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        self.expanded_indices = []
        for i in range(len(self.img_names)):
            for _ in range(self.points_per_image):
                self.expanded_indices.append(i)

    def __len__(self):
        return len(self.expanded_indices)

    def __getitem__(self, item):
        orig_idx = self.expanded_indices[item]
        img_name = self.img_names[orig_idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil, dtype=np.uint8)
        except Exception as e:
            print(f"[Skipped] {img_path} | Reason: {str(e)}")
            return self.__getitem__((item + 1) % len(self))

        mask_out = np.zeros_like(mask, dtype=np.uint8)
        mask_out[mask == 38] = 1 # Cat
        mask_out[mask == 75] = 2 # Dog
        mask_out[mask == 255] = 0 # Border => background

        if self.transform:
            transformed = self.transform(image=image, mask=mask_out)
            image = transformed['image']
            mask_out = transformed['mask']
            new_h, new_w = mask_out.shape[-2:]
        
        valid_points = []
        for class_id in [0, 1, 2]:
            y_coords, x_coords = np.where(mask_out == class_id)
            if len(y_coords) > 0:
                valid_points.extend([(x, y) for x, y in zip(x_coords, y_coords)])
        
        if not valid_points:
            y_coords, x_coords = np.where(mask_out != 255)
            valid_points = [(x, y) for x, y in zip(x_coords, y_coords)]
        
        if not valid_points:
            x, y = random.randint(0, new_w - 1), random.randint(0, new_h - 1)
        else:
            idx = random.randint(0, len(valid_points) - 1)
            x, y = valid_points[idx]
            
        point_heatmap = np.zeros((new_h, new_w), dtype=np.float32)
        point_heatmap[y, x] = 1.0
        point_heatmap = cv2.GaussianBlur(point_heatmap, (13, 13), 3)
        if point_heatmap.max() > 0:
            point_heatmap = point_heatmap / point_heatmap.max()

        point_coords = np.array([x / new_w, y / new_h], dtype=np.float32)
        binary_mask = np.zeros_like(mask_out, dtype=np.uint8)
        binary_mask[(mask_out == mask_out[y, x])] = 1

        binary_mask = torch.from_numpy(binary_mask)
        point_heatmap = torch.from_numpy(point_heatmap).unsqueeze(0)
        point_coords_tensor = torch.from_numpy(point_coords)

        return image, point_heatmap, binary_mask.long(), point_coords_tensor

#################################### NETWORK ARCHITECTURE ########################################
# Using HFCLIPVisionEncoder and SqueezeExcite from A.3
# ...
class HFCLIPVisionEncoder(nn.Module):
    # (Same as in A.3)
class SqueezeExcite(nn.Module):
    # (Same as in A.3)

class PointBasedSegmentationModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", num_classes=1):
        super(PointBasedSegmentationModel, self).__init__()
        self.encoder = HFCLIPVisionEncoder(clip_model_name=clip_model_name)
        in_channels = self.encoder.hidden_dim
        self.point_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), # 224->112
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # 112->56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56->28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 28->14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True) #14 -> 16
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + 256, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            SqueezeExcite(in_channels)
        )
        self.dec4 = nn.Sequential( # 14x14 -> 28x28
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True), SqueezeExcite(512)
        )
        self.dec3 = nn.Sequential( # 28x28 -> 56x56
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True), SqueezeExcite(256)
        )
        self.dec2 = nn.Sequential( # 56x56 -> 112x112
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True), SqueezeExcite(128)
        )
        self.dec1 = nn.Sequential( # 112 -> 224
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True), SqueezeExcite(64)
        )
        self.final_upsample = nn.Upsample(size=(224, 224), mode="bilinear", align_corners=True)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x, point_heatmap):
        img_features, _ = self.encoder(x)
        point_features = self.point_encoder(point_heatmap)
        combined_features = torch.cat([img_features, point_features], dim=1)
        bottleneck = self.fusion(combined_features)
        d4 = self.dec4(bottleneck)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        out = self.final_upsample(d1)
        out = self.final_conv(out)
        return out
        
#################################### TRANSFORMS AND DATASETS ########################################
# (Using transforms from A.3)
# ...

#################################### METRICS & FUNCTIONS & TRAINING ########################################
# (Similar training loop, but adapted for the new dataset which yields heatmaps)
# ...
