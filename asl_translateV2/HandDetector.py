# Psudeocode for this function
# Firstly, receive the information from CameraReader
# -->Frames
# Pass it to yolo26n.pt to apply a bounding box to the hands
# -->Apply a 20% buffer to the bounding box in order to keep hands firmly in the visibility of the camera
# Return the bounded box coordinates on the frame

from ultralytics import YOLO 
import CameraReader

class HandDetector():
    def __init__(self):
        self.model = YOLO("yolo26n.pt")
    
    def get_box(self, frame):
        # The results is simply yolo claiming the frame for itself, verbose means do not give us a terminal entry
        # for it
        results = self.model(frame, verbose = False)
        # We are going to store the frames into boxes
        boxes = results[0].boxes
        # If the boxes have no length, the boxes are empty
        if len(boxes) == 0:
            return None
        else:
            # x1, y1 is 0,0 (always less than the variable2)
            # x2, y2 is any other point to form a square
            # we are mapping those ints into forming a box
            x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            # we calculate the width and height by subtracting large from small
            width = x2 - x1
            height = y2 - y1
            # We add a 20% padding to the width and height
            Xpadding = width * .2
            Ypadding = height * .2
            # We subtract the padding in the top left corner to expand it and add the padding to the bottom right
            # variable to make them bigger, + make sure it is an int to prevent CV from crashing while 
            # making the box
            return (int(x1 - Xpadding), int(y1 - Ypadding), int(x2 + Xpadding), int(y2 + Ypadding))

            #x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
            #return (x1, y1, x2, y2)   # no padding — matches how the dataset was actually collected