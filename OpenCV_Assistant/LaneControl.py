import numpy as np
from KeyPresses import HoldKey


class LaneControler:
    def __init__(self):
        self.a = 0x1E
        self.d = 0x20
        self.w = 0x11
    
    def gradient(self, line):
        x1, y1, x2, y2 = line
        if x2 == x1:
            return 999999999
        return (y2 - y1) / (x2 - x1)
    
    def lane_angle_degrees(self, line):
        return np.degrees(np.arctan(1/self.gradient(line))) * -1
    
    def steering_angle(self, left_lane, right_lane):
        left_angle = self.lane_angle_degrees(left_lane)
        right_angle = self.lane_angle_degrees(right_lane)
        return (left_angle + right_angle) / 2
    
    def lane_control(self, left_lane, right_lane, steering):
        angle = self.steering_angle(left_lane, right_lane)
        time = 0.05 * abs(angle)/20
        
        if angle > 10:
            if steering:
                HoldKey(self.d, time)
                HoldKey(self.a, time/2)
            return angle
            
        elif angle < -10:
            if steering:
                HoldKey(self.a, time)
                HoldKey(self.d, time/2)
            return angle
        return angle
        
        

