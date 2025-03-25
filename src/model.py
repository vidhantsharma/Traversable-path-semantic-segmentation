import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
import torch.optim as optim

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

class UNetResNet(nn.Module):
    def __init__(self, num_classes=2):
        """
        U-Net with ResNet encoder for semantic segmentation.
        """
        super(UNetResNet, self).__init__()
        
        # Pretrained ResNet as encoder
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.encoder2 = resnet.layer1  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels
        
        # Decoder (Upsampling with correct channel dimensions)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Output: 256 channels
        self.dec4 = self.conv_block(512, 256)  # (256 up + 256 skip) → 256
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)  # (128 up + 128 skip) → 128
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)   # (64 up + 64 skip) → 64
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(96, 32)    # (64 up + 32 skip) → 32
        
        # Final segmentation layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        """ Basic convolutional block with BatchNorm and ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """ Forward pass with skip connections """
        # Encoder
        x1 = self.encoder1(x)  # 64 channels
        x2 = self.encoder2(x1)  # 64 channels
        x3 = self.encoder3(x2)  # 128 channels
        x4 = self.encoder4(x3)  # 256 channels
        x5 = self.encoder5(x4)  # 512 channels
        
        # Decoder
        x = self.upconv4(x5)  # 256 channels
        x = torch.cat([x, x4], dim=1)  # (256 + 256)
        x = self.dec4(x)  # Output: 256
        
        x = self.upconv3(x)  # 128 channels
        x = torch.cat([x, x3], dim=1)  # (128 + 128)
        x = self.dec3(x)  # Output: 128
        
        x = self.upconv2(x)  # 64 channels
        x = torch.cat([x, x2], dim=1)  # (64 + 64)
        x = self.dec2(x)  # Output: 64
        
        x = self.upconv1(x)  # 32 channels
        x1_upsampled = F.interpolate(x1, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1_upsampled], dim=1)
        # x = torch.cat([x, x1], dim=1)  # (32 + 32)
        x = self.dec1(x)  # Output: 32
        
        return self.final_conv(x)

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
        
def get_optimizer_and_scheduler(model, lr=1e-3, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return optimizer, scheduler
