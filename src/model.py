import cv2
import mediapipe as mp
import numpy as np

from utils import calculate_angle


class ModelPose:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.counter = 0
        self.stage= None

    def run(self,min_detection_confidence=0.5,min_tracking_confidence=0.5):


        
        return