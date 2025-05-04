import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import SimpleSegmentationCNN, UNetResNet, SegFormerModel
import os
from evaluation.evaluation_methods import EvaluationMethods

'''
This file will only test a single image output at a given time.
To get the combined metrics for all the test images, check evaluate.py
'''

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
# model = SimpleSegmentationCNN().to(device)
# model = UNetResNet().to(device)
model = SegFormerModel().to(device)

# Load the checkpoint
checkpoint_path = r"checkpoints\segformer_model.pth"  # Change this to your checkpoint path
# checkpoint_path = r"checkpoints\simple_cnn.pth"  # Change this to your checkpoint path
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # Ensure correct key
model.eval()

# Load and preprocess the test image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise Exception(f"Error: Image file '{image_path}' not found.")
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Error: Failed to load the image. Check the file format and path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (512, 512))  # Resize to match model input size
    img = img / 255.0  # Scale to [0, 1]
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

    # Apply normalization
    img = normalize(img)

    img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
    return img

# Load test image
image_path = r"processed_dataset\train\bremen\bremen_000015_000019\bremen_000015_000019_leftImg8bit.png"
gt_path = r"processed_dataset\train\bremen\bremen_000015_000019\binary_mask.png"
output_mask_path = r"output\predicted_mask.png"  # Path to save predicted mask

input_tensor = preprocess_image(image_path)

# Run inference
with torch.no_grad():
    output = model(input_tensor)  # Get raw logits
    probabilities = F.softmax(output, dim=1)  # Convert to probabilities
    predicted_mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()  # Get class with highest probability

# Save predicted mask
os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
cv2.imwrite(output_mask_path, predicted_mask * 255)  # Save as a binary mask image
print(f"Predicted mask saved to: {output_mask_path}")

# Visualize results
def visualize_results(image_path, predicted_mask):
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (256, 256))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(predicted_mask, cmap="gray")
    axes[1].set_title("Predicted Segmentation Mask")
    axes[1].axis("off")

    plt.show()

def evaluate_metrics(gt_mask_path, predicted_mask_path):
    if not os.path.exists(gt_mask_path):
        print(f"Error: The file {gt_mask_path} does not exist.")
        return
    elif not os.path.exists(predicted_mask_path):
        print(f"Error: The file {predicted_mask_path} does not exist.")
        return

    print(f"Loading images from: {gt_mask_path} and {predicted_mask_path}")

    # Load the ground truth and prediction images
    gt_image = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_image = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

    if gt_image is None:
        print(f"Error: Failed to load the ground truth image from {gt_mask_path}.")
        return
    elif pred_image is None:
        print(f"Error: Failed to load the prediction image from {predicted_mask_path}.")
        return

    print(f"Images loaded successfully. Ground truth shape: {gt_image.shape}, Prediction shape: {pred_image.shape}")

    # Resize ground truth if shape is different from prediction
    if gt_image.shape != pred_image.shape:
        print(f"Resizing ground truth mask from {gt_image.shape} to {pred_image.shape}")
        gt_image = cv2.resize(gt_image, (pred_image.shape[1], pred_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create an evaluator instance
    evaluator = EvaluationMethods(gt_image, pred_image)

    # Print the evaluation metrics
    print(f"IoU: {evaluator.IoU_method}")
    print(f"Pixel Accuracy: {evaluator.pixel_accuracy}")
    # print(f"F1 Score: {evaluator.f1_score_accuracy}")


# Show results
visualize_results(image_path, predicted_mask)

evaluate_metrics(gt_path, output_mask_path)
