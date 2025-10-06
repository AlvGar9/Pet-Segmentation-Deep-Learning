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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPModel
# Assuming SegmentationModel and PetDataset are defined
# from model_definition import SegmentationModel
# from dataset_definition import PetDataset

# 2) SETUP: DEVICE
# --------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3) METRICS AND VISUALIZATION
# --------------------------------------------------------------
def colorize_mask(mask, border_color=(255, 255, 255)):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 0, 0)} # Green for Cat, Red for Dog
    for label, color in color_map.items():
        color_mask[mask == label] = color
    color_mask[mask == 255] = border_color # For original trimap borders
    return color_mask

def compute_iou_dice_acc(pred, gt, num_classes=3):
    if pred.size == 0: return float('nan'), float('nan'), float('nan')
    pixel_acc = (pred == gt).sum() / pred.size
    ious, dices = [], []
    for c in range(num_classes):
        intersection = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        pred_count, gt_count = (pred == c).sum(), (gt == c).sum()
        iou = intersection / union if union else float('nan')
        dice = 2 * intersection / (pred_count + gt_count) if (pred_count + gt_count) else float('nan')
        if not np.isnan(iou): ious.append(iou)
        if not np.isnan(dice): dices.append(dice)
    return (np.mean(ious) if ious else 0.0, np.mean(dices) if dices else 0.0, pixel_acc)

def visualize_predictions_with_metrics(model, dataset, device, num_samples=3, 
                                       mean=(0.48145466, 0.4578275, 0.40821073), 
                                       std=(0.26862954, 0.26130258, 0.27577711)):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    fig, axs = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1: axs = [axs] # Make it iterable
    
    metrics = {"iou": 0, "dice": 0, "acc": 0}
    valid = 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img, gt_mask = dataset[idx]
            img_inp = img.unsqueeze(0).to(device)
            logits = model(img_inp)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # De-normalize image for visualization
            img_np = img.permute(1, 2, 0).cpu().numpy() * std + mean
            img_np = np.clip(img_np, 0, 1)
            gt_mask_np = gt_mask.cpu().numpy()

            gt_color = colorize_mask(gt_mask_np)
            pred_color = colorize_mask(pred_mask)
            
            # Superimpose prediction
            superposed = img_np.copy()
            mask_area = pred_mask > 0
            # Ensure pred_color is float and in [0,1] range for blending
            superposed[mask_area] = 0.5 * img_np[mask_area] + 0.5 * (pred_color[mask_area] / 255.0)

            # Plotting
            axs[i][0].imshow(img_np)
            axs[i][0].set_title("Original Image")
            axs[i][1].imshow(gt_color)
            axs[i][1].set_title("Ground Truth")
            axs[i][2].imshow(pred_color)
            axs[i][2].set_title("Prediction")
            axs[i][3].imshow(superposed)
            axs[i][3].set_title("Superimposed")

            for j in range(4): axs[i][j].axis("off")
            
            iou, dice, acc = compute_iou_dice_acc(pred_mask, gt_mask_np)
            if not any(np.isnan([iou, dice, acc])):
                metrics["iou"] += iou
                metrics["dice"] += dice
                metrics["acc"] += acc
                valid += 1
                
    plt.tight_layout()
    plt.show()

    if valid:
        print(f"Avg IoU: {metrics['iou']/valid:.4f}, Avg Dice: {metrics['dice']/valid:.4f}, Avg Acc: {metrics['acc']/valid:.4f}")
    else:
        print("No valid samples for metric calculation.")

# 4) SETUP: DATASET AND MODEL
# --------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

val_transform = A.Compose([
    A.LongestMaxSize(224),
    A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

# Assuming PetDataset is your dataset class
full_dataset = PetDataset(root="path/dataset", is_train=True, transform=val_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

model = SegmentationModel(num_classes=3).to(device)
state_dict = torch.load("path/to/your/model.pt", map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)
model.eval()

visualize_predictions_with_metrics(model, val_dataset, device=device, num_samples=5)
