from torch.utils.data import DataLoader
from src.dataloader import TraversablePathDataloader
from src.model import SegFormerModel
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from evaluation.evaluation_methods import EvaluationMethods
import cv2
import numpy as np

'''
This file will give average metrics values on validation data.
GT for test data is not available, so it was not used for finding average metrics.
To get the image wise results and corresponding prediction visualization, check test.py
'''

# Load test dataset
preprocess_data = False
BATCH_SIZE = 4

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAW_DATA_PATH = "raw_dataset"
PROCESSED_DATA_PATH = "processed_dataset"
CHECKPOINT_PATH = r"checkpoints\segformer_model.pth"

# Data Transforms
input_image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

target_mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Load Data
data_loader = TraversablePathDataloader(
    raw_data_path=RAW_DATA_PATH,
    processed_data_path=PROCESSED_DATA_PATH,
    batch_size=BATCH_SIZE,
    preprocess_data=preprocess_data,
    input_image_transform=input_image_transform,
    target_mask_transform=target_mask_transform,
    num_workers=2
)

test_data_loader = data_loader.get_validation_dataloader()

# Load model
model = SegFormerModel().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def main():
    # Evaluation loop
    total_iou = 0.0
    total_accuracy = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, (images, gt_masks) in enumerate(test_data_loader):
            if batch_idx % 10 == 0:
                print(f"Step [{batch_idx}/{len(test_data_loader)}]")
            images = images.to(DEVICE)
            gt_masks = gt_masks.squeeze(1).cpu().numpy()  # shape: (B, H, W)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # shape: (B, H, W)

            for gt_mask, pred_mask in zip(gt_masks, preds):
                gt_mask_bin = (gt_mask > 0).astype(np.uint8)
                pred_mask_bin = (pred_mask > 0).astype(np.uint8)

                evaluator = EvaluationMethods(gt_mask_bin, pred_mask_bin)
                total_iou += evaluator.IoU_method
                total_accuracy += evaluator.pixel_accuracy
                count += 1

    # Average results
    avg_iou = total_iou / count
    avg_acc = total_accuracy / count

    print("\nEvaluation Summary:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Pixel Accuracy: {avg_acc:.4f}")

if __name__ == "__main__":
    main()
