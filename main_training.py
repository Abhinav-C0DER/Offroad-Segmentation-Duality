import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp

# --- CONFIGURATION ---
DATASET_ROOT_DIR = "/kaggle/working/Offroad_Segmentation_Training_Dataset"
BACKBONE_SIZE = "base"
# ⚠️ NEW FILENAME TO PROTECT YOUR OLD MODELS
SAVE_PATH = "/kaggle/working/models/strategy3_warm_restarts.pth"

# ============================================================================
# Mask Utils
# ============================================================================
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
n_classes = len(value_map)

# ============================================================================
# Dataset
# ============================================================================
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask_mapped = np.zeros_like(mask, dtype=np.uint8)
        for raw, new in value_map.items(): mask_mapped[mask == raw] = new
        if self.transform:
            augmented = self.transform(image=image, mask=mask_mapped)
            image = augmented['image']
            mask_mapped = augmented['mask']
        return image, mask_mapped.long()

# ============================================================================
# Model
# ============================================================================
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.GELU(),
            nn.Conv2d(128, 128, 1), nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# --- STRATEGY 2/3: WEIGHTED LOSS ---
# We keep the aggressive weights from Strategy 2 because they worked!
weights = [0.1, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 0.1]
class_weights = torch.tensor(weights).float().cuda()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multiclass')
        self.focal = smp.losses.FocalLoss(mode='multiclass')
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, y_pred, y_true):
        return self.dice(y_pred, y_true) + self.focal(y_pred, y_true) + self.ce(y_pred, y_true)

# ============================================================================
# Main
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Strategy 3 (Warm Restarts) on: {device}")

    # Hyperparams
    batch_size = 8
    lr = 1e-4
    n_epochs = 30 # Perfect duration for 2 cycles
    w, h = 518, 518 

    # Augmentations (Slightly stronger rotation to prevent overfitting)
    train_transform = A.Compose([
        A.Resize(h, w), A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.4), # Increased rotation probability
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2(),
    ])
    
    train_loader = DataLoader(MaskDataset(os.path.join(DATASET_ROOT_DIR, 'train'), transform=train_transform), 
                              batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Loading DINOv2 {BACKBONE_SIZE}...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval().to(device)
    
    # Get Embed Dim
    sample = torch.randn(1, 3, h, w).to(device)
    with torch.no_grad(): n_embed = backbone.forward_features(sample)["x_norm_patchtokens"].shape[2]
    
    classifier = SegmentationHeadConvNeXt(n_embed, n_classes, w//14, h//14).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr)

    # --- NEW: Warm Restarts Scheduler ---
    # Cycle 1: 10 Epochs (Fast Exploration)
    # Cycle 2: 20 Epochs (Deep Exploitation)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Train
    print(f"Starting Training for {n_epochs} epochs with SGDR...")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    best_loss = float('inf')

    for epoch in range(n_epochs):
        classifier.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.no_grad(): features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(features)
            output_masks = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            
            loss = criterion(output_masks, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step scheduler PER BATCH for Warm Restarts (standard practice)
            scheduler.step(epoch + (pbar.n / len(train_loader)))
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
            
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # Save checkpoint if loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Optional: Save "best so far"
            # torch.save(classifier.state_dict(), SAVE_PATH.replace(".pth", "_best.pth"))

    # SAVE FINAL MODEL
    torch.save(classifier.state_dict(), SAVE_PATH)
    print(f"✅ Strategy 3 Complete! Model saved to: {SAVE_PATH}")
    
    # Verify file exists
    if os.path.exists(SAVE_PATH):
        print(f"File confirmed: {os.path.getsize(SAVE_PATH) / 1e6:.2f} MB")

if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT_DIR): main()