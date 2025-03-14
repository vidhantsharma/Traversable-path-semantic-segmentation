import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationCNN(nn.Module):
    def __init__(self, input_size=(256, 256)):
        """
        A simple fully convolutional neural network (FCN) for semantic segmentation.
        The model predicts a binary mask for traversable paths.

        Args:
            input_size (tuple): Expected input image size (H, W) to reconstruct ground truth mask.
        """
        super(SimpleSegmentationCNN, self).__init__()
        self.input_size = input_size  # Needed to reconstruct full mask from sparse GT

        # Encoder (Downsampling)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Decoder (Upsampling)
        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)  # Single channel output for binary mask

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input image tensor (batch_size, 3, H, W)

        Returns:
            Tensor: Output mask tensor (batch_size, 1, H, W)
        """
        # Encoder
        x1 = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)

        x2 = F.relu(self.conv2(x1))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)

        x3 = F.relu(self.conv3(x2))
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2)

        # Decoder
        x4 = F.relu(self.upconv1(x3))
        x5 = F.relu(self.upconv2(x4))

        # Output layer
        output = torch.sigmoid(self.final_conv(x5))  # Use sigmoid for binary mask

        return output


class SegmentationLoss(nn.Module):
    def __init__(self, input_size=(256, 256), use_dice_loss=True):
        """
        Custom loss function for semantic segmentation.

        Args:
            input_size (tuple): Expected input image size (H, W) for mask reconstruction.
            use_dice_loss (bool): Whether to use Dice loss along with BCE.
        """
        super(SegmentationLoss, self).__init__()
        self.input_size = input_size
        self.use_dice_loss = use_dice_loss
        self.bce_loss = nn.BCELoss()

    def reconstruct_mask(self, sparse_gt_dict, batch_size):
        """
        Converts sparse GT dictionary (linear indices of white pixels) to a full-size mask.

        Args:
            sparse_gt_dict (list of dicts): Each dict contains pixel indices for one sample.
            batch_size (int): Number of images in batch.

        Returns:
            Tensor: Full binary masks of shape (batch_size, 1, H, W).
        """
        height, width = self.input_size
        full_masks = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)

        for i in range(batch_size):
            indices = sparse_gt_dict[i]["white_pixel_indices"]  # Extract white pixel indices
            y_coords, x_coords = zip(*indices) if indices else ([], [])
            full_masks[i, 0, y_coords, x_coords] = 1.0  # Set white pixels to 1

        return full_masks.to(next(self.parameters()).device)  # Move to correct device (CPU/GPU)

    def dice_loss(self, preds, targets, smooth=1.0):
        """
        Computes the Dice loss for binary segmentation.

        Args:
            preds (Tensor): Model output (batch_size, 1, H, W).
            targets (Tensor): Ground truth masks (batch_size, 1, H, W).
            smooth (float): Smoothing term to avoid division by zero.

        Returns:
            Tensor: Dice loss value.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    def forward(self, predictions, sparse_gt_dict):
        """
        Compute the combined loss.

        Args:
            predictions (Tensor): Model output masks (batch_size, 1, H, W).
            sparse_gt_dict (list of dicts): Sparse ground truth masks as a dictionary.

        Returns:
            Tensor: Computed loss value.
        """
        batch_size = predictions.shape[0]
        full_gt_masks = self.reconstruct_mask(sparse_gt_dict, batch_size)

        # BCE Loss
        bce = self.bce_loss(predictions, full_gt_masks)

        # Dice Loss (Optional)
        if self.use_dice_loss:
            dice = self.dice_loss(predictions, full_gt_masks)
            return 0.5 * bce + 0.5 * dice  # Weighted combination
        else:
            return bce  # Only BCE loss
