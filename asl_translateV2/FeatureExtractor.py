# Pseudocode for this function
# Recieve box from HandDetector
# Recieve the raw frame from CameraReader
# Pass the box and frame to "Mediapipe" 
# Remove the 63 features from the frame
# Return the array 

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

class FeatureExtractor():
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            # We need two hands due to some gestures requiring 2 hands but this is for later
            num_hands=1,
            # The minimum confidence to even begin the translation
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def getfeature(self, frame, box):
        if box is None:
            # Fill box with zeroes to prevent fatal errors and fill empty boxes with zeroes rather than crashing 
            return None, None
        else:
            # Unfold the box to grab the coordinates
            x1, y1, x2, y2 = box
            # Grab the size of the frame
            frame_height, frame_width, _ = frame.shape
            # Prevent the frame from going out of the screen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            # Crop the image by the bounding box provided previously
            cropped_frame = frame[y1:y2, x1:x2]
            # CV operates as BGR not RGB, so the color conversion must apply
            colorcrop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            # Assign the new cropped image as a MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=colorcrop)
            # We then use the options to filter out anything that does not fit the confidence values
            mp_results = self.detector.detect(mp_image)
            # If mp_results.hand_landmarks cannot find a hand provide the box with zeroes
            if not mp_results.hand_landmarks:
                return None, None
            else:
                # Grab the first available frame that contains a hand 
                hand_landmarks = mp_results.hand_landmarks[0]
                extracted_points = []
                # Get size of whole picture and size of the cropped picture 
                frame_h, frame_w, _ = frame.shape
                crop_h, crop_w, _ = cropped_frame.shape
                # The wrist should always be 0 no matter what to provide proper subtraction math since distance 
                # from finger to wrist is similiar across all people
                wrist = hand_landmarks[0]
                # Wrist X; grab global coordinate and normalize it
                wrist_global_x = x1 + int(wrist.x * crop_w)
                wrist_norm_x = wrist_global_x / frame_w
                # Wrist Y; grab global coordinate and normalize it
                wrist_global_y = y1 + int(wrist.y * crop_h)
                wrist_norm_y = wrist_global_y / frame_h
                # Wrist z
                wrist_z = wrist.z
                # For every other coordinate in landmarks
                debug_points = []
                for landmark in hand_landmarks:
                    global_x = x1 + int(landmark.x * crop_w)
                    global_y = y1 + int(landmark.y * crop_h)
                    debug_points.append((global_x, global_y))
                    
                    # Normalize screen scale (0 to 1)
                    raw_norm_x = global_x / frame_w
                    raw_norm_y = global_y / frame_h
                    
                    # Subtract the Anchor
                    # This makes the wrist ALWAYS (0,0,0). 
                    final_x = raw_norm_x - wrist_norm_x
                    final_y = raw_norm_y - wrist_norm_y
                    final_z = landmark.z - wrist_z 

                    # Place the completed points into the created list
                    extracted_points.extend([final_x, final_y, final_z])
                return np.array(extracted_points), debug_points
            