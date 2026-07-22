# Pseudocode for this function
# Folder Search Logic
# This is specifically for letter training
# Hardcode folder name
#   For each Folder in letters A-Z
#       Open the folder, read the .npy file
#       Create Dataset
#       Have the Interpreter read the Dataset
#           Move Onto next folder
#       50 epochs, check if no improvements over 10 epochs
#       automatically stop
#       Store the .pth file

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ASL_Model import ASLSequenceInterpreter
from torch.utils.data import random_split

class ASLDataset(Dataset):
    def __init__(self,datapath):
            # to be moved to main datapath = 'C:\Users\night\Desktop\AmericanSignLanguageTranslator\asl_translate\ASL_Dataset'
            # 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
            self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            # the dataset is the index keeper for each letter, like the first .npy in A would be 0 and the last .npy of Z would be xxx index 
            self.dataset = []
            # Keep track of indexes for ever letter
            for label_index, letter in enumerate(self.letters):
                # Combine base path with the letter
                folder_path = os.path.join(datapath, letter)
                # Loop through every file in that specific letter's folder
                for file_name in os.listdir(folder_path):
                    # Get full path to the .npy file
                    file_path = os.path.join(folder_path,file_name)
                    # Save the package as a path and the matching number label
                    self.dataset.append((file_path, label_index))
    
    def __len__(self):
          return len(self.dataset)
    
    def __getitem__(self, index):
          file_path, label = self.dataset[index]
          numpy_data = np.load(file_path)
          tensor_data = torch.tensor(numpy_data, dtype = torch.float32)
          return (tensor_data, label)
    
datapath = r'C:\Users\night\Desktop\AmericanSignLanguageTranslator\asl_translateV2\ASL_Dataset'
dataset = ASLDataset(datapath)
model = ASLSequenceInterpreter()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split into train/val (80/20)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_loss = float('inf')
epochs_no_improve = 0
patience = 10

for epoch in range(50):
    # Train
    model.train()
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_data)
        loss = loss_function(predictions, batch_labels)
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_predictions = model(val_data)
            val_loss += loss_function(val_predictions, val_labels).item()
            correct += (val_predictions.argmax(dim=1) == val_labels).sum().item()
            total += val_labels.size(0)
    val_loss /= len(val_dataloader)
    val_accuracy = correct / total

    print(f'Epoch: {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.1f}%')

    # Checkpoint only if improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'Bestest_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch} (no improvement in {patience} epochs)')
            break