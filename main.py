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
from CameraReader import CameraReader
from HandDetector import HandDetector
from FeatureExtractor import FeatureExtractor
from LSTM_Interfacer import LSTM_Interfacer


camming = CameraReader()
detecting = HandDetector()
extracting = FeatureExtractor()
interfacing = LSTM_Interfacer()

while True:
    # Find a frame from CameraReader
    frame = camming.read()
    # If we cannot find a frame, try again
    if frame is None:
        continue
    # When a frame is found pass it to HandDetector to put a bounding box on it
    detected = detecting.get_box(frame)
    # Pass the bounding box and the frame to the FeatureExtractor to get feature from it
    extracted = extracting.getfeature(frame, detected)
    # Pass the extracted features to the LSTM_Interfacer to predict the gesture
    predicted = interfacing.predict(extracted)
    # Show the predicted letter
    print(predicted)
    cv2.imshow('ASL Translator', frame)
    # Push Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Close out of every window
cv2.destroyAllWindows()
# Turn off the camera
camming.stop()