# Pseudocode for this function
# Setup:
# Call CameraReader, HandDetector, FeatureExtractor, and the LSTM_Interfacer to turn them on and be ready for use
# Infinite Running Loop:
# Have CameraReader turn the camera on and fetch a frame
# Have that frame get passed to HandDetector to apply a 20% padded bounding box to the frame
# Have the bounding box and the frame get passed to FeatureExtractor to crop the image by the bounding box, 
# landmark it and pull the coordinates from the frame, relationally to the wrist, and return 63 extracted points
# After receiving the 63 extracted points from the FeatureExtractor, the LSTM_Interfacer classify the points 
# Against its weight path, to predict the correct gesture
# Safe Exit:
# Closing camera reader via the stop function which stops all passive functions down the pipeline

import cv2
import torch
from CameraReader import CameraReader
from HandDetector import HandDetector
from FeatureExtractor import FeatureExtractor
from LSTM_Interfacer import LSTM_Interfacer



camming = CameraReader(source=1)
detecting = HandDetector()
extracting = FeatureExtractor()
interfacing = LSTM_Interfacer()
# Translation key
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

while True:
    # Find a frame from CameraReader
    frame = camming.read()
    # If we cannot find a frame, try again
    if frame is None:
        continue
    # When a frame is found pass it to HandDetector to put a bounding box on it
    detected = detecting.get_box(frame)
    # Don't crash if the box doesn't exist
    if detecting.get_box is None: continue
    # Pass the bounding box and the frame to the FeatureExtractor to get feature from it
    extracted = extracting.getfeature(frame, detected)
    # Don't crash if there could not be features found
    if extracting.getfeature is None: continue
    # Pass the extracted features to the LSTM_Interfacer to predict the gesture
    predicted = interfacing.predict(extracted)
    # Upon early frames and uncler frames, dont let the system crash
    if predicted is None: continue
    # Find the index of the highest score (index 0 for A, index 1 for B)
    best_guess_index = torch.argmax(predicted).item()
    # Translate tensor data into a letter that is understood by the alphabet
    predicted_letter = alphabet[best_guess_index]
    if detected is not None:
        x1, y1, x2, y2 = detected
        cv2.rectangle(frame, (x1, y1), (x2,y2), (0,255,0), 2)
        print("Predicted letter: ", predicted_letter)
        cv2.putText(frame, predicted_letter, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow('ASL Translator', frame)
    # Push Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close out of every window
cv2.destroyAllWindows()
# Turn off the camera
camming.stop()