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
MODEL_PATH = "/kaggle/working/models/strategy3_warm_restarts.pth" 
BACKBONE_SIZE = "base"

# ============================================================================
# Metrics & Utils
# ============================================================================
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

def calculate_ap50(pred_logits, target, num_classes=10):
    """
    Calculates the Average Precision at 50% IoU threshold for each class.
    A class is considered a 'True Positive' if its IoU >= 0.5.
    """
    pred_labels = torch.argmax(pred_logits, dim=1) # (B, H, W)
    
    batch_ap50 = []
    class_aps = [[] for _ in range(num_classes)]
    
    for b in range(pred_labels.shape[0]):
        img_ious = []
        for cls in range(num_classes):
            pred_inds = (pred_labels[b] == cls)
            target_inds = (target[b] == cls)
            
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            
            if union == 0:
                img_ious.append(float('nan'))
            else:
                iou = (intersection / union).item()
                # AP50 logic: 1 if IoU >= 0.5, else 0
                ap_score = 1.0 if iou >= 0.5 else 0.0
                class_aps[cls].append(ap_score)
                img_ious.append(ap_score)
        
        batch_ap50.append(np.nanmean(img_ious))
        
    return np.nanmean(batch_ap50), [np.mean(c) if len(c) > 0 else float('nan') for c in class_aps]

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
# Main Loop
# ============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running mAP@50 Evaluation (Strategy 3 + TTA) on: {device}")
    
    # 1. Setup Data
    w, h = 518, 518
    val_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_loader = DataLoader(MaskDataset(os.path.join(DATASET_ROOT_DIR, 'val'), transform=val_transform), batch_size=4, shuffle=False)
    
    # 2. Load Model
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval().to(device)
    
    sample = torch.randn(1, 3, h, w).to(device)
    with torch.no_grad(): n_embed = backbone.forward_features(sample)["x_norm_patchtokens"].shape[2]
    
    classifier = SegmentationHeadConvNeXt(n_embed, 10, w//14, h//14).to(device)
    classifier.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    classifier.eval()
    
    # 3. Evaluation
    total_mAP = []
    class_cumulative_ap = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # TTA: Pass 1
            feat1 = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits1 = classifier(feat1)
            
            # TTA: Pass 2 (Flip)
            imgs_flip = torch.flip(imgs, dims=[3])
            feat2 = backbone.forward_features(imgs_flip)["x_norm_patchtokens"]
            logits2 = classifier(feat2)
            logits2 = torch.flip(logits2, dims=[3])
            
            # Combine
            avg_logits = (logits1 + logits2) / 2.0
            preds = F.interpolate(avg_logits, size=(h, w), mode="bilinear", align_corners=False)
            
            mAP, cAPs = calculate_ap50(preds, masks)
            total_mAP.append(mAP)
            class_cumulative_ap.append(cAPs)

    # 4. Report
    final_mAP50 = np.nanmean(total_mAP)
    print("\n" + "="*35)
    print(f"🏆 FINAL mAP@50 (TTA): {final_mAP50:.4f}")
    print("="*35)
    
    avg_class_ap = np.nanmean(class_cumulative_ap, axis=0)
    for idx, name in enumerate(class_names):
        print(f"{name:15}: {avg_class_ap[idx]:.4f}")

if __name__ == "__main__":
    main()