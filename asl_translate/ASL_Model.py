"""
ASL_Model.py
TL:DR This program serves as the model for which the entire backbone of the ASL program runs off of. An LSTM (Long Short-Term Memory) receives the batches of .npy files and classes them down per video of letter.
It contains a 30% drop-out system is also implemented so it does not memorize the data.
"""

import torch
import torch.nn as nn

#Ultimate goal is to evolve this system from just spelling out static letters to recognizing continuous, 
#flowing sentences or multi-step gestures (like the full ASL sign for "I love you"), 
#the entire scope of the project expands.
#The abstraction levels would look like this:
#Layer 1: Handles the raw finger and wrist geometry over time.
#Layer 2: Identifies the individual micro-gestures or standalone letters.
#Layer 3: Acts as the sentence parser, understanding the context between those letters and gestures to string them together into complete words.
#Upgrade ASL_Model.py to three layers when it is time to retrain on your massive new dataset.


class ASLSequenceInterpreter(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, num_classes=26):
        super(ASLSequenceInterpreter, self).__init__()
        
        # THE MEMORY LAYER LSTM (LongShortTermMemory) 
        # input_size = 63 (because 21 MediaPipe joints * 3 coordinates (x,y,z))
        # hidden_size = 128 is default (the number of "neurons" parsing the patterns)
        # Layer amount = how many LSTM's the code runs through (one finds basic motion patterns, one uses basic motions to find complex gestures)
        # batch_first=True tells PyTorch data is structured as (Batch comes first then Sequence then Features)
        # Batch = How many videos the model looks at simultaneously through training
        # Sequence = How many frames are in each video
        # Features = How many coordinates it needs to see
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # THE DECISION LAYER
        # Processes the sequence, and force the model to make a guess
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU() # Activation function (keeps the math non-linear)
        self.dropout = nn.Dropout(0.3) # Prevents the LSTM from just memorizing the data
        
       # 64 internal features squashed down to the 26 classes
        self.fc2 = nn.Linear(64, num_classes) 

    def forward(self, x):
        # x is input data coming in. Shape: (batch_size, 16 frames, 63 points)
        
        # Pass the 16 frames through the LSTM
        out, (hn, cn) = self.lstm(x)
        
        # The LSTM gives us an output for ALL 16 frames. 
        # Only the network's final conclusion matters so slice it out after seeing the 16th frame.
        # Give me all batches, only the very last frame (index -1), and all 128 hidden features of that frame
        out = out[:, -1, :] 
        
        # Pass the conclusion through the decision layers
        #fc1 takes 128 features and compresses them to 64
        out = self.fc1(out)
        # relu is max between (0,x) which turns all negative numbers to zero, 
        # which helps the network learn non-linear, complex relationships instead of just drawing 
        # straight lines through data
        out = self.relu(out)
        # dropout randomly turns off 30% of features to force the network to learn what "A" is rather than 
        # memorize coordinate locations 
        out = self.dropout(out)
        #fc2 push 64 features into the last 26 features (letters)
        out = self.fc2(out)
        
        # Returns an array of probabilities for each letter
        return out
    
