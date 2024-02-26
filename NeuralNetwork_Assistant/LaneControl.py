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
            return 99999999
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
        
    def lane_control_mean(self, slope, steering):
        time = 0.05 * abs(slope)
        angle = np.degrees(np.arctan(1/slope)) * -1
        
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

    def lane_control_barriers(self, new_angle, old_angle, intersect, steering):
        time = 0.05 * abs(new_angle)/40
        middle = 128
        border = 40
        steering_angle = new_angle
        
        # if abs(new_angle - old_angle) > 90:
        #     steering_angle = old_angle
        # else:
        #     steering_angle = new_angle
        #     old_angle = np.copy(new_angle)
        
        # if line is central
        if middle - border < intersect < middle + border: 
            if steering_angle > 10:
                steering_direction = "right steer"
                if steering:
                    HoldKey(self.d, 2*time)
                    HoldKey(self.a, time)
                
            elif steering_angle < -10:
                steering_direction = "left steer"
                if steering:
                    HoldKey(self.a, 2*time)
                    HoldKey(self.d, time)
            else: 
                steering_direction = "straight"
                
        elif intersect <= middle - border:
            if abs(steering_angle) < 30:
                steering_direction = "left shift"
                if steering:
                    HoldKey(self.a, time)
                    HoldKey(self.d, time/2)
            else:
                if steering_angle < 0:
                    steering_direction = "sharp right"
                    if steering:
                        HoldKey(self.d, time)
                        HoldKey(self.a, time/2)
                else:
                    steering_direction = "sharp left (u)"
                    if steering:
                        HoldKey(self.a, time)
                        HoldKey(self.d, time/2)
                        
                
        elif middle + border <= intersect:
            if  abs(steering_angle) < 30:
                steering_direction = "right shift"
                if steering:
                    HoldKey(self.d, time)
                    HoldKey(self.a, time/2)
            else: 
                if 0 < steering_angle:
                    steering_direction = "sharp right"
                    if steering:
                        HoldKey(self.d, time)
                        HoldKey(self.a, time/2)
                else:
                    steering_direction = "sharp left (u)"
                    if steering:
                        HoldKey(self.a, time)
                        HoldKey(self.d, time/2)
                    
                
        else: 
            steering_direction = "straight"
        
        return steering_angle, steering_direction