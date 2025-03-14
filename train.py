import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from src.dataloader import TraversablePathDataloader
from src.model import SimpleSegmentationCNN, SegmentationLoss
import os

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAW_DATA_PATH = "raw_dataset"
PROCESSED_DATA_PATH = "processed_dataset"
LOG_DIR = "logs/segmentation"
SAVE_MODEL_PATH = "checkpoints/segmentation_model.pth"

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load Data
print("Loading dataset...")
data_loader = TraversablePathDataloader(
    raw_data_path=RAW_DATA_PATH,
    processed_data_path=PROCESSED_DATA_PATH,
    batch_size=BATCH_SIZE,
    preprocess_data=False,
    transform=transform,
    num_workers=4
)

train_loader = data_loader.get_train_dataloader()
val_loader = data_loader.get_validation_dataloader()

# Model, Loss, Optimizer
model = SimpleSegmentationCNN().to(DEVICE)
criterion = SegmentationLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
writer = SummaryWriter(LOG_DIR)

# Training Loop
def train():
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_idx, (images, sparse_masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            
            # Move sparse masks to CPU for processing, then back to device
            sparse_masks = [{k: v.to("cpu") for k, v in mask.items()} for mask in sparse_masks]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, sparse_masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        
        validate(epoch)
        
        torch.save(model.state_dict(), SAVE_MODEL_PATH)

# Validation Loop
def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, sparse_masks in val_loader:
            images = images.to(DEVICE)
            sparse_masks = [{k: v.to("cpu") for k, v in mask.items()} for mask in sparse_masks]
            outputs = model(images)
            loss = criterion(outputs, sparse_masks)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    writer.add_scalar("Loss/Validation", avg_loss, epoch)
    print(f"Validation Loss after Epoch {epoch+1}: {avg_loss:.4f}")

if __name__ == "__main__":
    train()
    writer.close()
