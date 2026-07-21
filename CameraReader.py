# Pseudocode for the function
# find the camera, usb check or hardware check
# Lets use CV2 for that
# Ask for user permission to use the camera
# Turn on the camera
# --> If no close the program and give the user an error stating that the program requires camera permissions
# Remember user permismissions maybe through a .data file, and where the camera is to increase speed of start
# Input from the camera needs to be read
# If the input hiccups, wait for next non-hiccuped frame
# --> if continous, inform the user about frame lag, and possiblity of incorrect data predictions
# Prior to whenever the ASLTranslator closes, turn off camera

import cv2
import threading

class CameraReader():

    def __init__(self, source =0):
        self.cap = cv2.VideoCapture(source)
        self.running = True
        self.frame = None
        self.thread = threading.Thread(target = self._update, daemon = True)
        self.thread.start()

    def _update(self):
        #Keep updating so long as the system is running
        while self.running == True:
            success, frame = self.cap.read()
            #Can we find the frame? if not try again for the next frame
            if success == True:
                #Provide us the found frame
                self.frame = frame

    def read(self):
        #Hand over the most recent frame
        return self.frame

    def stop(self):
        #Stop the next frame from being collected
        self.running = False
        #Put any currently running frame into the thread
        self.thread.join()
        #Safely close the camera
        self.cap.release() 