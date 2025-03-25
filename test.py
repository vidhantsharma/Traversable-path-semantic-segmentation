import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import SimpleSegmentationCNN
import os
from evaluation.evaluation_methods import EvaluationMethods

# Load the tra$ined model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSegmentationCNN().to(device)

# Load the checkpoint
checkpoint_path = "checkpoints\segmentation_model.pth"  # Change this to your checkpoint path
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load and preprocess the test image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise Exception(f"Error: Image file '{image_path}' not found.")
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Error: Failed to load the image. Check the file format and path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (256, 256))  # Resize to match model input size
    img = img / 255.0  # Normalize to [0,1]
    img = np.transpose(img, (2, 0, 1))  # Change from (H, W, C) to (C, H, W)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    return img

# Load test image
image_path = r"processed_dataset\val\frankfurt\frankfurt_000000_000294\frankfurt_000000_000294_leftImg8bit.png"
gt_path = r"processed_dataset\val\frankfurt\frankfurt_000000_000294\binary_mask.png"
input_tensor = preprocess_image(image_path)

# Run inference
with torch.no_grad():
    output = model(input_tensor)  # Get raw logits
    probabilities = F.softmax(output, dim=1)  # Convert to probabilities
    predicted_mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()  # Get class with highest probability

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

def calculate_metrics(gt_mask, predicted_mask):
    evaluator = EvaluationMethods(gt_mask, predicted_mask)
    iou = evaluator.IoU_method
    pixel_acc = evaluator.pixel_accuracy
    f1_score = evaluator.f1_score_accuracy
    return iou, pixel_acc, f1_score

# Show results
visualize_results(image_path, predicted_mask)

# find metrics
if not os.path.exists(gt_image_path):
    print(f"Error: The file {gt_image_path} does not exist.")
elif not os.path.exists(pred_image_path):
    print(f"Error: The file {pred_image_path} does not exist.")
else:
    print(f"Loading images from: {gt_image_path} and {pred_image_path}")

    # Load the ground truth and prediction images
    gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    pred_image = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)

    if gt_image is None:
        print(f"Error: Failed to load the ground truth image from {gt_image_path}.")
    elif pred_image is None:
        print(f"Error: Failed to load the prediction image from {pred_image_path}.")
    else:
        print(f"Images loaded successfully. Ground truth shape: {gt_image.shape}, Prediction shape: {pred_image.shape}")

        # Ensure the images have the same shape
        if gt_image.shape != pred_image.shape:
            print("Error: The ground truth and prediction images must have the same shape.")
        else:
            # Create an evaluator instance
            evaluator = EvaluationMethods(gt_image, pred_image)

            # Print the evaluation metrics
            print(f"IoU: {evaluator.IoU_method}")
            print(f"Pixel Accuracy: {evaluator.pixel_accuracy}")
            print(f"F1 Score: {evaluator.f1_score_accuracy}")

