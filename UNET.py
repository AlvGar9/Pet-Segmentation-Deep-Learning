# -*- coding: utf-8 -*-
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
class PetDataset(Dataset):
    def __init__(self, root="path/dataset", is_train=True, transform=None):
        self.transform = transform
        self.classes = ['background', 'cat', 'dog', 'border']
        self.root = os.path.abspath(root)
        path = "TrainVal" if is_train else "Test"
        self.image_dir = os.path.join(self.root, path, "color")
        self.mask_dir = os.path.join(self.root, path, "label")
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"Check dataset structure. Missing: {self.image_dir} or {self.mask_dir}")
        self.img_names = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        img_name = self.img_names[item]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        # Ensure file exists
        if not os.path.exists(img_path):
            print(f"Missing image file: {img_path}. Skipping...")
            return self.__getitem__((item + 1) % len(self.img_names))
        if not os.path.exists(mask_path):
            print(f"Missing mask file: {mask_path}. Skipping...")
            return self.__getitem__((item + 1) % len(self.img_names))
        try:
            # Load the image
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Corrupted image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Load the mask
            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil, dtype=np.uint8)
        except Exception as e:
            print(f"Skipping corrupt file: {img_path} | Error: {e}")
            return self.__getitem__((item + 1) % len(self.img_names)) # Skip and try next
        # Convert from dataset's mask values to [0,1,2,...]
        mask_out = np.zeros_like(mask, dtype=np.uint8)
        mask_out[mask == 38] = 1 # Cat
        mask_out[mask == 75] = 2 # Dog
        mask_out[mask == 255] = 0 # Border => ignored class
        if self.transform:
            transformed = self.transform(image=image, mask=mask_out)
            image = transformed['image']
            mask_out = transformed['mask']
        if isinstance(mask_out, np.ndarray):
            mask_out = torch.from_numpy(mask_out)
        return image, mask_out.long()

#################################### NETWORK ARCHITECTURE ########################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Double convolution layer with ReLU activation.
        Parameters:
        - in_channels: int, number of input channels.
        - out_channels: int, number of output channels.
        """
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Downsampling layer with double convolution and max pooling.
        Parameters:
        - in_channels: int, number of input channels.
        - out_channels: int, number of output channels.
        """
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upsampling layer with transpose convolution and double convolution.
        Parameters:
        - in_channels: int, number of input channels.
        - out_channels: int, number of output channels.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        UNet architecture for image segmentation.
        Parameters:
        - in_channels: int, number of input channels.
        - num_classes: int, number of output classes.
        """
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.bottle_neck = DoubleConv(512, 1024)
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        b = self.bottle_neck(p4)
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        out = self.out(up_4)
        return out

#################################### TRANSFORMS AND DATASETS ########################################
train_transform = A.Compose([
    # A.Resize(224, 224),
    # Step 1: Resize maintaining aspect ratio, largest side becomes 224
    A.LongestMaxSize(max_size=224),
    # Step 2: Pad remaining space to ensure final dimensions 224x224
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
    # A.Resize(224, 224),
    # Step 1: Resize maintaining aspect ratio, largest side becomes 224
    A.LongestMaxSize(max_size=224),
    # Step 2: Pad remaining space to ensure final dimensions 224x224
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

full_train_dataset = PetDataset(is_train=True, transform=train_transform)
train_ratio = 0.8
train_size = int(train_ratio * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

#################################### METRICS & FUNCTIONS ########################################
dice_metric = DiceScore(num_classes=3).to(device)
iou_metric = MeanIoU(num_classes=3).to(device)
acc_metric = MulticlassAccuracy(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss(ignore_index = 255)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader, desc="Training", leave=True):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_segmentation_metrics(model, dataloader, device):
    model.eval()
    dice_metric.reset()
    iou_metric.reset()
    acc_metric.reset()
    running_loss = 0.0
    total_samples = len(dataloader.dataset)
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=True):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images) # (B, C, H, W)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1) # (B, H, W)
            # ** Fix ignored pixels **
            valid_mask = masks != 255 # Boolean mask for valid pixels
            preds = preds[valid_mask] # Remove ignored pixels
            masks = masks[valid_mask] # Remove ignored pixels
            # ** Ensure valid data before proceeding **
            if preds.numel() == 0 or masks.numel() == 0:
                continue # Skip batch if there's no valid data
            # ** One-hot encoding **
            preds_onehot = F.one_hot(preds, num_classes=3).permute(1, 0).unsqueeze(0).int() # Convert to int64
            masks_onehot = F.one_hot(masks, num_classes=3).permute(1, 0).unsqueeze(0).int() # Convert to int64
            # ** Compute Metrics (Ensure correct shape and type) **
            dice_metric.update(preds_onehot.float(), masks_onehot.float()) # DiceScore needs float
            iou_metric.update(preds_onehot, masks_onehot) # ** Fix: Uses integer masks **
            acc_metric.update(preds, masks) # Accuracy works with raw class indices

    avg_loss = running_loss / total_samples
    dice_val = dice_metric.compute().item()
    iou_val = iou_metric.compute().item()
    acc_val = acc_metric.compute().item()
    return {
        "val_loss": avg_loss,
        "dice": dice_val,
        "iou": iou_val,
        "acc": acc_val
    }

####################################
# train_eval_once function
####################################
def train_eval_once(
    lr,
    batch_size,
    weight_decay=1e-3,
    patience=5,
    max_epochs=30,
    device=device
):
    """
    Train + evaluate the model once using given hyperparameters.
    Returns (best_val_loss, best_model_state).
    """
    # Re-build dataset / dataloaders with given batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize a fresh model
    model = UNet(in_channels=3, num_classes=3).to(device)
    model = torch.nn.DataParallel(model) # Enable multi-GPU
    model.to(device)

    # New optimizer with the given LR + WD
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    # Lists to record curves
    train_losses_run = []
    val_losses_run = []
    dice_run = []
    iou_run = []
    acc_run = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate_segmentation_metrics(model, val_loader, device)
        val_loss = metrics["val_loss"]
        dicemetric = metrics["dice"]
        ioumetric = metrics["iou"]
        accmetric = metrics["acc"]

        train_losses_run.append(train_loss)
        val_losses_run.append(val_loss)
        dice_run.append(metrics["dice"])
        iou_run.append(metrics["iou"])
        acc_run.append(metrics["acc"])

        print(f"LOSSES FOR THIS EPOCH | TRAIN LOSS: {train_loss}, VAL LOSS: {val_loss}")
        print(f"METRICS FOR THIS EPOCH | DICE: {dicemetric}, IOU: {ioumetric}, ACC: {accmetric}")
        print()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} for LR={lr}, WD={weight_decay}.")
            break

    curves = {
        "train_losses": train_losses_run,
        "val_losses": val_losses_run,
        "dice": dice_run,
        "iou": iou_run,
        "acc": acc_run
    }
    
    torch.save(best_model_state, f"seg_model_{lr}_{batch_size}.pt")
    return best_val_loss, best_model_state, curves

####################################
# Hyperparameter Search (example cases)
####################################
# Initialize tracking variables
best_overall_loss = float("inf")
best_hparams = None
best_state = None

# Case 1
learning_rate = 1e-2
batch_size = 4
# ... (repeat for all hyperparameter combinations)

# Example for the best found hyperparameters
learning_rate = 1e-3 
batch_size = 64
print(f"\n=== Running with LR={learning_rate}, BS={batch_size} ===")
val_loss, model_state, curves = train_eval_once(
    lr=learning_rate,
    batch_size=batch_size,
    patience=5,
    max_epochs=30,
    device=device
)
print(f"Final best val_loss={val_loss:.4f} for LR={learning_rate}, BS={batch_size}")
if val_loss < best_overall_loss:
    best_overall_loss = val_loss
    best_hparams = (learning_rate, batch_size)
    best_state = model_state

# Save the performance plot
plot_filename = f"training_performance_lr{learning_rate}_bs{batch_size}.png"
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Plot Loss Curves
axes[0].plot(range(1, len(curves["train_losses"]) + 1), curves["train_losses"],
             label="Train Loss", marker="o")
axes[0].plot(range(1, len(curves["val_losses"]) + 1), curves["val_losses"],
             label="Val Loss", marker="o")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True)

# Plot Metric Curves
axes[1].plot(range(1, len(curves["dice"]) + 1), curves["dice"],
             label="Dice Score", marker="o")
axes[1].plot(range(1, len(curves["iou"]) + 1), curves["iou"],
             label="Mean IoU", marker="o")
axes[1].plot(range(1, len(curves["acc"]) + 1), curves["acc"],
             label="Pixel Accuracy", marker="o")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Metric")
axes[1].set_title("Performance Metrics")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(plot_filename)
plt.show()
plt.close()

print(f"Saved performance plot for LR={learning_rate}, BS={batch_size} as '{plot_filename}'")
print("\nHyperparameter search complete!")
print(f"Best val_loss={best_overall_loss:.4f} with LR={best_hparams[0]}, BS={best_hparams[1]}")

####################################
# Save and evaluate the best model
####################################
from collections import OrderedDict

# Remove 'module.' prefix from keys if they exist
new_state_dict = OrderedDict()
for k, v in best_state.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

# Load the corrected state dict into your model
final_best_model = UNet(in_channels=3, num_classes=3).to(device)
final_best_model.load_state_dict(new_state_dict)

# Save the corrected model state
torch.save(final_best_model.state_dict(), "best_segmentation_model.pt")
print("Best model weights saved to best_segmentation_model.pt")

################Evaluation on test set###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=3).to(device)

state_dict = torch.load("path/model", map_location=device)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict, strict=False)
model.eval()

test_transform = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])

test_dataset = PetDataset(is_train=False, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
metrics = evaluate_segmentation_metrics(model, test_loader, device)
test_loss = metrics["val_loss"]
dicemetric = metrics["dice"]
ioumetric = metrics["iou"]
accmetric = metrics["acc"]

print(f"LOSSES FOR THIS EPOCH | TEST LOSS: {test_loss}")
print(f"METRICS FOR THIS EPOCH | DICE: {dicemetric}, IOU: {ioumetric}, ACC: {accmetric}")
print()
