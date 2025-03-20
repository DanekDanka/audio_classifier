import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torchmetrics import F1Score
import kagglehub

# Download dataset
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

# Configuration
MAX_LENGTH = 100
N_MFCC = 40
SR = 22050
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess dataset 
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio, _ = librosa.load(self.file_paths[idx], sr=SR)
        mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        
        # Pad/truncate
        if mfccs.shape[1] < MAX_LENGTH:
            pad_width = MAX_LENGTH - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_LENGTH]
            
        # Convert to tensor and reshape to (channels, sequence_length)
        mfccs = torch.FloatTensor(mfccs)  # Shape: (40, 100)
        label = torch.LongTensor([self.labels[idx]])
        return mfccs, label

# CNN model
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=N_MFCC, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 8)
        )
        
    def forward(self, x):
        return self.net(x)

# Collect file paths and labels
file_paths = []
labels = []
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.wav'):
            parts = file.split('-')
            emotion = int(parts[2]) - 1  # 0-based
            file_paths.append(os.path.join(root, file))
            labels.append(emotion)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    file_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Create dataloaders
train_dataset = AudioDataset(X_train, y_train)
test_dataset = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = EmotionCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
f1_metric = F1Score(task="multiclass", num_classes=8).to(DEVICE)

best_val_acc = 0.0
best_model_path = "model/best_model.pth"

# Train model
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())
    
    train_loss = running_loss / len(train_loader)
    val_loss = test_loss / len(test_loader)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean()
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
    print("-" * 50)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with val_acc: {best_val_acc:.4f}")

# Save final model
final_model_path = "final_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# Load model
# model = EmotionCNN().to(DEVICE)
# model.load_state_dict(torch.load(best_model_path))
# model.eval()

# Final test evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)
        
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu())
        all_labels.extend(labels.cpu())

final_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean()
final_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\nFinal Test Accuracy: {final_acc:.4f}")
print(f"Final Test F1 Score: {final_f1:.4f}")
