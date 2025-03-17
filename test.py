import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.model import SimpleSegmentationCNN
import os

# Load the trained model
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
image_path = r"processed_dataset\train\aachen\aachen_000162_000019\aachen_000162_000019_leftImg8bit.png"
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

# Show results
visualize_results(image_path, predicted_mask)
