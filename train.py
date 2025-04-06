import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from src.dataloader import TraversablePathDataloader
from src.model import SimpleSegmentationCNN, UNetResNet, SegNet,SegmentationLoss
import os

# Params
preprocess_data = False

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAW_DATA_PATH = "raw_dataset"
PROCESSED_DATA_PATH = "processed_dataset"
LOG_DIR = "logs/segmentation"
CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Load Data
data_loader = TraversablePathDataloader(
    raw_data_path=RAW_DATA_PATH,
    processed_data_path=PROCESSED_DATA_PATH,
    batch_size=BATCH_SIZE,
    preprocess_data=preprocess_data,
    transform=transform,
    num_workers=2
)

train_loader = data_loader.get_train_dataloader()
val_loader = data_loader.get_validation_dataloader()

# Model, Loss, Optimizer
# model = SimpleSegmentationCNN().to(DEVICE)
# model = UNetResNet().to(DEVICE)
model = SegNet().to(DEVICE)
criterion = SegmentationLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
writer = SummaryWriter(LOG_DIR)

# Load latest checkpoint if exists
def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth") and "best_checkpoint" not in f]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(CHECKPOINT_DIR, f)))
    return os.path.join(CHECKPOINT_DIR, latest_checkpoint)

def load_checkpoint():
    checkpoint_path = get_latest_checkpoint()
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Loaded checkpoint from {checkpoint_path} (epoch {start_epoch})")
            return start_epoch, best_val_loss
        except:
            return 0, float("inf")
    return 0, float("inf")

# Training Loop with Early Stopping
def train(use_checkpoint=False):
    start_epoch, best_val_loss = load_checkpoint() if use_checkpoint else (0, float("inf"))
    patience_counter = 0
    
    for epoch in range(start_epoch, NUM_EPOCHS):
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
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"segmentation_checkpoint_epoch_{epoch+1}.pth")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved at {checkpoint_path}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(checkpoint, BEST_CHECKPOINT_PATH)
            print(f"✅ Best model improved! Saved at {BEST_CHECKPOINT_PATH}.")
        else:
            patience_counter += 1
            print(f"🔸 No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs!")
            break

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
    train(use_checkpoint=False)
    writer.close()
