# Pseudocode for this function
    #Folder logic
    #check name of folder
    #if name of folder =
        #cv2.VideoCapture('pathtovideo.mp4')
        #framed = save 16 frames
    #pass framed into detecting
    #pass detected into extracting
    #save extracted to another folder
#To be worked on for words, not for letters

from HandDetector import HandDetector
from FeatureExtractor import FeatureExtractor
import os

detecting = HandDetector()
extracting = FeatureExtractor()

while True:
    pass