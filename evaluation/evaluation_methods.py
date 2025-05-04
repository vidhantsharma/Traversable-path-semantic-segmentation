import cv2
import numpy as np
import os
from sklearn.metrics import f1_score

class EvaluationMethods:
    def __init__(self, ground_truth, prediction):
        self.ground_truth = ground_truth
        self.prediction = prediction
    
    @property
    def IoU_method(self):
        intersection = np.logical_and(self.ground_truth, self.prediction)
        union = np.logical_or(self.ground_truth, self.prediction)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 1.0
        return iou

    @property
    def pixel_accuracy(self):
        correct_pixels = np.sum(self.ground_truth == self.prediction)
        total_pixels = self.ground_truth.size
        accuracy = correct_pixels / total_pixels
        return accuracy
    
    @property
    def f1_score_accuracy(self):
        gt_flat = self.ground_truth.flatten().astype(np.uint8)
        pred_flat = self.prediction.flatten().astype(np.uint8)
        return f1_score(gt_flat, pred_flat, average='binary')

# if __name__ == "__main__":
#     # Paths to the ground truth and prediction images
#     gt_image_path = r"C:\Users\Pragv\OneDrive\Desktop\ML\Traversable-path-semantic-segmentation\binary_mask.png"
#     pred_image_path = r"C:\Users\Pragv\OneDrive\Desktop\ML\Traversable-path-semantic-segmentation\predicted_binary_mask.png"
    
#     # Check if the files exist
#     if not os.path.exists(gt_image_path):
#         print(f"Error: The file {gt_image_path} does not exist.")
#     elif not os.path.exists(pred_image_path):
#         print(f"Error: The file {pred_image_path} does not exist.")
#     else:
#         print(f"Loading images from: {gt_image_path} and {pred_image_path}")
    
#         # Load the ground truth and prediction images
#         gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
#         pred_image = cv2.imread(pred_image_path, cv2.IMREAD_GRAYSCALE)
    
#         if gt_image is None:
#             print(f"Error: Failed to load the ground truth image from {gt_image_path}.")
#         elif pred_image is None:
#             print(f"Error: Failed to load the prediction image from {pred_image_path}.")
#         else:
#             print(f"Images loaded successfully. Ground truth shape: {gt_image.shape}, Prediction shape: {pred_image.shape}")
    
#             # Ensure the images have the same shape
#             if gt_image.shape != pred_image.shape:
#                 print("Error: The ground truth and prediction images must have the same shape.")
#             else:
#                 # Create an evaluator instance
#                 evaluator = EvaluationMethods(gt_image, pred_image)
    
#                 # Print the evaluation metrics
#                 print(f"IoU: {evaluator.IoU_method}")
#                 print(f"Pixel Accuracy: {evaluator.pixel_accuracy}")
#                 print(f"F1 Score: {evaluator.f1_score_accuracy}")
