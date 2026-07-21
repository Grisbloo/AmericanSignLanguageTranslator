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

class ASLDataset(Dataset):
    def __init__(self,datapath):
           # Folders we are looking for
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

datapath = r'C:\Users\night\Desktop\AmericanSignLanguageTranslator\asl_translate\ASL_Dataset'
dataset = ASLDataset(datapath)
dataloader = DataLoader(dataset, batch_size=32, shuffle = True)
model = ASLSequenceInterpreter()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epochs in range(50):
    for batch_data, batch_labels in dataloader:
          optimizer.zero_grad()
          predictions = model(batch_data)
          loss = loss_function(predictions, batch_labels)
          loss.backward()
          optimizer.step()
    print('Epoch num: ', epochs, 'Loss amount: ', loss.item())
torch.save(model.state_dict(), 'Bestest_model.pth')