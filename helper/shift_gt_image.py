import cv2
import numpy as np
import os

def upshift_image(image, shift_pixels):
    # Create an empty array with the same shape as the input image
    upshifted_image = np.zeros_like(image)
    
    # Check if shift_pixels is greater than 0 and less than the height of the image
    if shift_pixels > 0 and shift_pixels < image.shape[0]:
        # Shift the image up by the specified number of pixels
        upshifted_image[:-shift_pixels] = image[shift_pixels:]
    elif shift_pixels == 0:
        # If shift_pixels is 0, return the original image
        upshifted_image = image
    else:
        print(f"Error: shift_pixels ({shift_pixels}) is greater than or equal to the image height ({image.shape[0]}).")
    
    return upshifted_image

def save_image(image, path):
    cv2.imwrite(path, image)

if __name__ == "__main__":
    # Path to the ground truth image
    gt_image_path = r"C:\Users\Pragv\OneDrive\Desktop\ML\Traversable-path-semantic-segmentation\binary_mask.png"
    
    # Check if the file exists
    if not os.path.exists(gt_image_path):
        print(f"Error: The file {gt_image_path} does not exist.")
    else:
        print(f"Loading image from: {gt_image_path}")
    
        # Load the ground truth image
        gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
    
        if gt_image is None:
            print(f"Error: Failed to load the image from {gt_image_path}.")
        else:
            print(f"Image loaded successfully. Image shape: {gt_image.shape}")
    
            # Calculate the number of pixels to shift (20% of the image height)
            shift_pixels = int(0.0 * gt_image.shape[0])
            print(f"Shifting image up by {shift_pixels} pixels.")
    
            # Upshift the image by the calculated number of pixels
            upshifted_image = upshift_image(gt_image, shift_pixels)
            print(f"Upshifted image shape: {upshifted_image.shape}")
    
            # Path to save the modified upshifted image
            output_path = r"C:\Users\Pragv\OneDrive\Desktop\ML\Traversable-path-semantic-segmentation\predicted_binary_mask.png"
    
            # Save the upshifted image
            save_image(upshifted_image, output_path)
            print(f"Upshifted image saved at: {output_path}")