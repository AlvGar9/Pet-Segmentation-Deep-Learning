#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import MulticlassAccuracy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Assuming SegmentationModel is defined elsewhere
# from model_definition import SegmentationModel

################################
# 1) DEVICE SETUP
################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

################################
# 2) LOAD YOUR BEST MODEL WEIGHTS
################################
model = SegmentationModel(num_classes=3).to(device)
state_dict = torch.load("path/to/your/model.pt", map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

################################
# 3) DEFINE A DATASET FOR THE TEST SPLIT
################################
class PetTestDataset(Dataset):
    def __init__(self, root="path/dataset"):
        super().__init__()
        self.root = os.path.abspath(root)
        self.image_dir = os.path.join(self.root, "Test", "color")
        self.mask_dir = os.path.join(self.root, "Test", "label")
        self.img_names = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        try:
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_pil = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_pil, dtype=np.uint8)
            mask_out = np.zeros_like(mask_arr, dtype=np.uint8)
            mask_out[mask_arr == 38] = 1
            mask_out[mask_arr == 75] = 2
            mask_out[(mask_arr == 255) | (mask_arr == 0)] = 0
            return image, mask_out
        except Exception as e:
            print(f"Skipping file '{img_path}' due to: {e}")
            next_idx = (idx + 1) % len(self.img_names)
            return self.__getitem__(next_idx)

test_dataset = PetTestDataset(root="path/dataset")

################################
# 4) DEFINE A SET OF IMAGE PERTURBATIONS
################################
def add_gaussian_noise(image, std):
    if std <= 0: return image
    noise = np.random.randn(*image.shape) * std
    out = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def apply_blur(image, times):
    if times <= 0: return image
    out = image.copy()
    for _ in range(times):
        out = cv2.blur(out, (5,5))
    return out

def change_contrast(image, factor):
    out = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return out

def change_brightness(image, shift):
    out = np.clip(image.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    return out

def occlude_square(image, square_size):
    if square_size <= 0: return image
    out = image.copy()
    h, w, _ = out.shape
    y0 = np.random.randint(0, max(1, h-square_size+1))
    x0 = np.random.randint(0, max(1, w-square_size+1))
    out[y0:y0+square_size, x0:x0+square_size, :] = 0
    return out

def add_salt_and_pepper(image, amount):
    if amount <= 0: return image
    out = image.copy()
    h, w, _ = out.shape
    num_pix = int(amount*h*w)
    # Salt
    ys = np.random.randint(0, h, num_pix)
    xs = np.random.randint(0, w, num_pix)
    out[ys, xs, :] = 255
    # Pepper
    ys = np.random.randint(0, h, num_pix)
    xs = np.random.randint(0, w, num_pix)
    out[ys, xs, :] = 0
    return out

################################
# 5) EVALUATION PIPELINE
################################
post_transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

dice_metric = DiceScore(num_classes=3).to(device)
iou_metric = MeanIoU(num_classes=3).to(device)
acc_metric = MulticlassAccuracy(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()

def evaluate_dataset(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    dice_metric.reset(), iou_metric.reset(), acc_metric.reset()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            preds = torch.argmax(logits, dim=1)
            preds_1hot = F.one_hot(preds, num_classes=3).permute(0, 3, 1, 2).float()
            masks_1hot = F.one_hot(masks, num_classes=3).permute(0, 3, 1, 2).float()
            dice_metric.update(preds_1hot, masks_1hot)
            iou_metric.update(preds_1hot.int(), masks_1hot.int())
            acc_metric.update(preds, masks)
    
    return {
        "loss": total_loss / total_count,
        "dice": dice_metric.compute().item(),
        "iou": iou_metric.compute().item(),
        "acc": acc_metric.compute().item()
    }

class PerturbedDataset(Dataset):
    def __init__(self, base_dataset, perturb_fn, param, post_transform):
        self.base_ds, self.perturb_fn, self.param, self.post_transform = base_dataset, perturb_fn, param, post_transform
    def __len__(self):
        return len(self.base_ds)
    def __getitem__(self, idx):
        raw_img, raw_mask = self.base_ds[idx]
        perturbed = self.perturb_fn(raw_img, self.param)
        transformed = self.post_transform(image=perturbed, mask=raw_mask)
        return transformed['image'], transformed['mask'].long()

################################
# 6) PERTURBATION EXPERIMENTS
################################
def perturbation_experiments(test_dataset, model, device):
    # This function would loop through different perturbation functions and parameters,
    # create a PerturbedDataset for each, evaluate it, and plot the results.
    # Example for one perturbation type:
    stds = np.linspace(0, 20, 10)
    dice_scores = []
    for std in stds:
        perturbed_ds = PerturbedDataset(test_dataset, add_gaussian_noise, std, post_transform)
        metrics = evaluate_dataset(model, perturbed_ds, device)
        dice_scores.append(metrics['dice'])
        print(f"Gaussian Noise (std={std:.2f}): Dice = {metrics['dice']:.4f}")
    
    plt.plot(stds, dice_scores, marker='o')
    plt.title('Dice Score vs. Gaussian Noise')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Mean Dice Score')
    plt.grid(True)
    plt.show()

# Run evaluation on clean dataset first
clean_dataset = PerturbedDataset(test_dataset, lambda x, param: x, 0, post_transform)
clean_metrics = evaluate_dataset(model, clean_dataset, device)
print("Clean Test Set Metrics:", clean_metrics)

# Run all perturbation experiments
perturbation_experiments(test_dataset, model, device)
print("Robustness evaluation completed!")
