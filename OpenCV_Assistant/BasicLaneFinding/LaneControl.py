import numpy as np
from pyautogui import keyDown, keyUp
from time import sleep
import cv2

class LaneControler:
    def __init__(self):
        pass 
    
    def gradient(self, line):
        x1, y1, x2, y2 = line
        if x2 == x1:
            return 0
        return (y2 - y1) / (x2 - x1)
    
    def lane_angle_degrees(self, line):
        return np.degrees(np.arctan(self.gradient(line))) * -1
    
    def steering_angle(self, left_lane, right_lane):
        left_angle = self.lane_angle_degrees(left_lane)
        right_angle = self.lane_angle_degrees(right_lane)
        return (left_angle + right_angle) / 2
    
    def lane_control(self, left_lane, right_lane):
        angle = self.steering_angle(left_lane, right_lane)
        time = 0.05
        if angle > 10:
            keyDown("d")
            sleep(time)
            keyUp("d")
            return "R", angle
            
        elif angle < -10:
            keyDown("a")
            sleep(time)
            keyUp("a")
            return "L", angle
        return "S", angle
        
        
        
        
        