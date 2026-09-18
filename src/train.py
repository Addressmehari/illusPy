import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from src.model import ShapeCNN

class GenericShapeDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform

        pos_files = glob.glob(os.path.join(pos_dir, "*.png"))
        for fpath in pos_files:
            self.samples.append(fpath)
            self.labels.append(1.0)

        neg_files = glob.glob(os.path.join(neg_dir, "*.png"))
        for fpath in neg_files:
            self.samples.append(fpath)
            self.labels.append(0.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(fpath).convert("L")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

def train_shape_model(shape="square", base_dir="dataset", models_dir="models", epochs=25):
    """Trains a CNN model to classify the specified shape."""
    pos_dir = os.path.join(base_dir, f"{shape}s")
    neg_dir = os.path.join(base_dir, f"non_{shape}s")

    if not os.path.exists(pos_dir) or not os.path.exists(neg_dir):
        print(f"Error: Dataset directories '{pos_dir}' or '{neg_dir}' do not exist!")
        return

    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, f"{shape}_cnn.pth")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = GenericShapeDataset(pos_dir, neg_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = ShapeCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Training CNN for shape: '{shape.upper()}' ({len(dataset)} samples, {epochs} epochs) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = train_loss / total
        epoch_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_labels = val_labels.unsqueeze(1)
                val_outputs = model(val_images)
                val_preds = (torch.sigmoid(val_outputs) >= 0.5).float()
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

    torch.save(model.state_dict(), model_save_path)
    # Also save to root for backwards compatibility
    torch.save(model.state_dict(), f"{shape}_cnn.pth")
    print(f"Model training complete! Weights saved to '{model_save_path}'.")
