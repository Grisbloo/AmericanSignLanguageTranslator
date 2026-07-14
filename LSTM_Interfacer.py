# Pseudocode for this function
# Turn on the LSTM
# Pass the array of extracted points to the LSTM
# Hold onto a running 16 frame memory sored in a list   
# Provide the 16 frames to the LSTM for classification
# Return the predicted letter with confidence of prediction

import torch
from ASL_Model import ASLSequenceInterpreter

class LSTM_Interfacer():
    def __init__(self):
        # Create LSTM 16 frame memory
        self.LSTMMemory = []
        # Use the Model we have created prior
        self.model = ASLSequenceInterpreter()
        # Use the weights created from training
        self.model.load_state_dict(torch.load('best_asl_model.pth'))
        # Use the model's evaluation mode so there is no dropout
        self.model.eval()

    def predict(self, features):
        # Add landmarked frames to the LSTM's memory bank
        self.LSTMMemory.append(features)
        # If the LSTM's memory is over 16 remove the oldest element
        if len(self.LSTMMemory) > 16:
            self.LSTMMemory.pop(0)
        # If the LSTM's memory is 16 we can use it
        if len(self.LSTMMemory) == 16:
            # The LSTM needs a specific datatype 
            input_tensor = torch.tensor(self.LSTMMemory, dtype=torch.float32)
            # Adds 1 to the batch number due to our program needing 3 things (Batch, Sequence, Features)
            input_tensor = input_tensor.unsqueeze(0)
            # Don't waste CPU power on learning due to a selected weighted path being used
            with torch.no_grad():
                output = self.model(input_tensor)
            return output
        # If the LSTM's memory is not full don't start computing until it gets to 16
        return None
    