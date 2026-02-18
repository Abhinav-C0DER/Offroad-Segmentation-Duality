import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT_DIR = "/kaggle/working/Offroad_Segmentation_Training_Dataset"
# Pointing to your NEW Strategy 3 model
MODEL_PATH = "/kaggle/working/models/strategy3_warm_restarts.pth" 
BACKBONE_SIZE = "base"

# ============================================================================
# Utils & Dataset
# ============================================================================
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0: ious.append(float('nan'))
        else: ious.append((intersection/union).item())
    return np.nanmean(ious), ious

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

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 128, 7, padding=3), nn.GELU())
        self.block = nn.Sequential(nn.Conv2d(128, 128, 7, padding=3, groups=128), nn.GELU(), nn.Conv2d(128, 128, 1), nn.GELU())
        self.classifier = nn.Conv2d(128, out_channels, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

# ============================================================================
# Evaluation
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Evaluating Strategy 3 with TTA on: {device}")
    
    # 1. Setup Data
    w, h = 518, 518
    val_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_loader = DataLoader(MaskDataset(os.path.join(DATASET_ROOT_DIR, 'val'), transform=val_transform), batch_size=4, shuffle=False)
    
    # 2. Load Models
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval().to(device)
    
    sample = torch.randn(1, 3, h, w).to(device)
    with torch.no_grad(): n_embed = backbone.forward_features(sample)["x_norm_patchtokens"].shape[2]
    
    classifier = SegmentationHeadConvNeXt(n_embed, 10, w//14, h//14).to(device)
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.eval()
    
    # 3. Inference Loop
    all_ious = []
    class_ious = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Pass 1: Original
            feat1 = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits1 = classifier(feat1)
            
            # Pass 2: Horizontal Flip
            imgs_flip = torch.flip(imgs, dims=[3])
            feat2 = backbone.forward_features(imgs_flip)["x_norm_patchtokens"]
            logits2 = classifier(feat2)
            logits2 = torch.flip(logits2, dims=[3]) # Flip back
            
            # Average & Rescale
            avg_logits = (logits1 + logits2) / 2.0
            preds = F.interpolate(avg_logits, size=(h, w), mode="bilinear", align_corners=False)
            
            mean_iou, cls_iou = compute_iou(preds, masks)
            all_ious.append(mean_iou)
            class_ious.append(cls_iou)

    # 4. Final Report
    print("\n" + "="*35)
    print(f"🏆 STRATEGY 3 TTA mIoU: {np.nanmean(all_ious):.4f}")
    print("="*35)
    
    avg_cls_iou = np.nanmean(class_ious, axis=0)
    for idx, name in enumerate(class_names):
        print(f"{name:15}: {avg_cls_iou[idx]:.4f}")

if __name__ == "__main__":
    main()