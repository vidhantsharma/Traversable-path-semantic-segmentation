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
PATIENCE = 5  # Stop training if val loss doesn't improve for 'PATIENCE' epochs
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

# Training Loop with Early Stopping
def train():
    print("Starting training...")
    
    best_val_loss = float("inf")  # Track best validation loss
    patience_counter = 0  # Count epochs without improvement

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        avg_val_loss = validate(epoch)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience
            torch.save(model.state_dict(), SAVE_MODEL_PATH)  # Save best model
            print(f"✅ Model improved! Saved at epoch {epoch+1}.")
        else:
            patience_counter += 1
            print(f"🔸 No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs!")
            break  # Stop training if no improvement for 'PATIENCE' epochs

# Validation Loop
def validate(epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    writer.add_scalar("Loss/Validation", avg_loss, epoch)
    print(f"Validation Loss after Epoch {epoch+1}: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
    train()
    writer.close()
