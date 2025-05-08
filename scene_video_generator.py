import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from src.model import SimpleSegmentationCNN, UNetResNet, SegNet, SegFormerModel
import torch
import torch.nn.functional as F
from torchvision import transforms

class SceneVideoGenerator:
    def __init__(self, scene_dir, output_path="output.mp4", checkpoint_path=None, model=None ,fps=10):
        self.scene_dir = scene_dir
        self.output_path = output_path
        self.fps = fps
        self.image_files = sorted(glob(os.path.join(scene_dir, "*.png")))  # or use *.jpg if needed
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {scene_dir}")

        self.model = self.load_model(model=model, checkpoint_path=checkpoint_path)
        self.frame_size = self._get_frame_size()

    def load_model(self, checkpoint_path = None, model = None):
        print("Loading model...")

        model.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict']) 
        model.eval()

        return model

    def predict_mask(self, image_path):
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
        img = self.normalize(img)

        img = img.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

        # Run inference
        with torch.no_grad():
            output = self.model(img)  # Get raw logits
            probabilities = F.softmax(output, dim=1)  # Convert to probabilities
            predicted_mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()  # Get class with highest probability

        return predicted_mask

    def _get_frame_size(self):
        return (512 * 2, 512)  # (width, height) for side-by-side: original + mask

    def generate_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        for img_path in tqdm(self.image_files, desc=f"Generating video from {self.scene_dir}"):
            mask = self.predict_mask(image_path=img_path)
            image = cv2.imread(img_path)
            # Downsample image to mask size
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
            # Convert binary mask to a blue overlay
            color_mask = np.zeros_like(image)
            color_mask[mask == 1] = [255, 0, 0]  # Blue for class 1

            # Optional: blend with original image
            blended = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
            combined = np.hstack((image, blended))
            out.write(combined)

        out.release()
        print(f"✅ Video saved to: {self.output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate side-by-side video for Cityscapes scene")
    parser.add_argument("--scene", default=r"leftImg8bit_demoVideo\leftImg8bit\demoVideo\stuttgart_01",help="Path to scene folder (e.g. ./cityscapes/stuttgart_00)")
    parser.add_argument("--output", default="output.mp4", help="Output video filename")
    parser.add_argument("--checkpoint_path", default=r"checkpoints\segformer.pth" ,help="model checkpoint path")

    # model = SimpleSegmentationCNN()
    # model = UNetResNet()
    # model = SegNet()
    model = SegFormerModel() # Change this to the model of checkpoint path

    args = parser.parse_args()

    generator = SceneVideoGenerator(scene_dir=args.scene, output_path=args.output, checkpoint_path=args.checkpoint_path, model=model)
    generator.generate_video()
