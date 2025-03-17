import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationCNN(nn.Module):
    def __init__(self):
        """
        U-Net inspired CNN for binary semantic segmentation.
        """
        super(SimpleSegmentationCNN, self).__init__()

        # Encoder (Downsampling)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Decoder (Upsampling with correct input channels)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(96, 32, kernel_size=3, padding=1)  # Adjust for concatenation (32 + 32)
        
        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(48, 16, kernel_size=3, padding=1)  # Adjust for concatenation (16 + 16)

        self.upconv3 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Conv2d(24, 8, kernel_size=3, padding=1)  # Adjust for concatenation (8 + 8)

        self.final_conv = nn.Conv2d(8, 2, kernel_size=1)  # 2 output channels for binary segmentation

    def forward(self, x):
        """
        Forward pass with skip connections.
        """
        # Encoder
        x1 = F.relu(self.conv1(x))
        x1_pooled = F.max_pool2d(x1, kernel_size=2, stride=2)  # 128x128

        x2 = F.relu(self.conv2(x1_pooled))
        x2_pooled = F.max_pool2d(x2, kernel_size=2, stride=2)  # 64x64

        x3 = F.relu(self.conv3(x2_pooled))
        x3_pooled = F.max_pool2d(x3, kernel_size=2, stride=2)  # 32x32

        # Decoder with skip connections
        x4 = F.relu(self.upconv1(x3_pooled))  # 64x64
        x4 = torch.cat([x4, x3], dim=1)  # Skip connection (now 64 channels)
        x4 = F.relu(self.conv4(x4))  # Reduce back to 32 channels

        x5 = F.relu(self.upconv2(x4))  # 128x128
        x5 = torch.cat([x5, x2], dim=1)  # Skip connection (now 32 channels)
        x5 = F.relu(self.conv5(x5))  # Reduce back to 16 channels

        x6 = F.relu(self.upconv3(x5))  # 256x256
        x6 = torch.cat([x6, x1], dim=1)  # Skip connection (now 16 channels)
        x6 = F.relu(self.conv6(x6))  # Reduce back to 8 channels

        output = self.final_conv(x6)  # Logits
        return output

class SegmentationLoss(nn.Module):
    def __init__(self, use_dice_loss=True):
        """
        Loss function for segmentation.
        Args:
            use_dice_loss (bool): Whether to use Dice loss along with CrossEntropy.
        """
        super(SegmentationLoss, self).__init__()
        self.use_dice_loss = use_dice_loss
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, preds, targets, smooth=1.0):
        """
        Computes the Dice loss for binary segmentation.
        Args:
            preds (Tensor): Model output logits (batch_size, 2, H, W)
            targets (Tensor): Ground truth (batch_size, H, W)
        Returns:
            Tensor: Dice loss value.
        """
        preds = torch.softmax(preds, dim=1)[:, 1, :, :]  # Class 1 (traversable) probability

        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1).float()  # Convert to float for division
        
        intersection = (preds * targets).sum()

        return 1 - (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    def forward(self, predictions, targets):
        """
        Compute loss.
        Args:
            predictions (Tensor): Model output logits (batch_size, 2, H, W).
            targets (Tensor): Target masks (batch_size, 1, H, W) with values {0,1}.
        Returns:
            Tensor: Computed loss.
        """
        # Move targets to same device as predictions
        targets = targets.to(predictions.device)

        # Convert (batch_size, 1, H, W) → (batch_size, H, W)
        targets = targets.squeeze(1).long()

        # Compute cross-entropy loss (logits should be passed directly)
        ce = self.ce_loss(predictions, targets)

        if self.use_dice_loss:
            dice = self.dice_loss(predictions, targets)
            return 0.5 * ce + 0.5 * dice  # Weighted loss
        else:
            return ce  # Only CE loss
