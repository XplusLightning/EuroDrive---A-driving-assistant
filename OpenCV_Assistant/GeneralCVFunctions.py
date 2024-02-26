import cv2
import numpy as np
from mss import mss 
from PIL import Image
from math import sin, cos, radians

class GeneralFunctions:
    def __init__(self):
        pass 
    
    def resize(self, image, scale_percent=50):
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        new_dimensions = (width, height)
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA) 
    
    
    def rectangle_mask(self, image, vertices):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, vertices[0], vertices[1], 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    
    def polygon_mask(self, image, vertices):
        vertices = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(image, image, mask=mask)
    
    
    def greyscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    def blur(self, image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    
    def edge_detection(self, image, threshold_1=50, threshold_2=150):
        return cv2.Canny(image, threshold_1, threshold_2)

    
    def approximate_lane_lines(self, image):
        lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, minLineLength=50, maxLineGap=500)
        if lines is None:
            return [0, 0, 0, 0], [0, 0, 0, 0], np.zeros_like(image)
        left_lane, right_lane = self.lane_approximations(image, lines)
        lane_image = self.draw_lines(image, lines, (8, 118, 241, 1))
        return left_lane, right_lane, lane_image
    
    def draw_lines(self, image, lines, colour=(0, 255, 0, 1)):
        blank = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_GRAY2BGRA)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4) 
                if x1 != x2:
                    if abs((y2-y1)/(x2-x1)) > 0.2:
                        cv2.line(blank, (x1, y1), (x2, y2), colour, 2)
                        
        return blank
        
        
    def scaled_array(self, array, scale):
        new_array = []
        for i in array:
            new_array.append((int(i[0] * scale/100), int(i[1] * scale/100)))
        return new_array
    
    def adjusted_array(self, array, adjustment):
        new_array = []
        new_array.append((int(array[0][0] - adjustment), int(array[0][1] - adjustment)))
        new_array.append((int(array[1][0] + adjustment), int(array[1][1] - adjustment)))
        new_array.append((int(array[2][0] + adjustment), int(array[2][1] + adjustment)))
        new_array.append((int(array[3][0] - adjustment), int(array[3][1] + adjustment)))
        return new_array
    
    def lane_approximations(self, image, lines):
        if len(lines) == 0:
            return [0, 0, 0, 0], [0, 0, 0, 0]
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x1 != x2:
                slope, y_intercept = np.polyfit((x1, x2), (y1, y2), 1)
                slope_threshold = 0.2
                if slope < -1 * slope_threshold:
                    left_fit.append((slope, y_intercept))
                elif slope > slope_threshold:
                    right_fit.append((slope, y_intercept))

        if len(left_fit) != 0:
            left_fit_average = np.average(left_fit, axis=0)
            left_line = self.make_coordinates(image, left_fit_average)
        else:
            left_line = [0, 0, 0, 0]
            
        if len(right_fit) != 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.make_coordinates(image, right_fit_average)
        else:
            right_line = [0, 0, 0, 0]
        
        return left_line, right_line
    
    
    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0] - 250
        y2 = int(y1 * 0.7)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return [x1, y1, x2, y2]
                    
    def screenshot(self):
        bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        sct = mss()
        sct_img = sct.grab(bounding_box)
        frame = Image.frombytes(
                'RGB', 
                (sct_img.width, sct_img.height), 
                sct_img.rgb, 
            )
        return cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    
    def lane_gui(self, image, angle):
        x, y = 0, 0  
        if angle > 10:
            foreground = cv2.cvtColor(cv2.imread("OpenCV_Assistant\BasicLaneFinding\direction_images\\right.png"), cv2.COLOR_BGR2BGRA)
            rows, cols, channels = foreground.shape
            image[y:y+rows, x:x+cols] = foreground
        
        elif angle < -10:
            foreground = cv2.cvtColor(cv2.imread("OpenCV_Assistant\BasicLaneFinding\direction_images\left.png"), cv2.COLOR_BGR2BGRA)
            cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
            rows, cols, channels = foreground.shape
            image[y:y+rows, x:x+cols] = foreground        
            
        else:
            foreground = cv2.cvtColor(cv2.imread("OpenCV_Assistant\BasicLaneFinding\direction_images\straight.png"), cv2.COLOR_BGR2BGRA)
            cv2.cvtColor(foreground, cv2.COLOR_BGR2BGRA)
            rows, cols, channels = foreground.shape
            image[y:y+rows, x:x+cols] = foreground
        
        rad = 100
        point1 = (960, 800)
        point2 = (int(rad*sin(radians(angle))+960), 800-int(rad*cos(radians(angle))))
        image = cv2.arrowedLine(image, point1, point2, (0, 0, 255, 1), 2)
        
        overlayed_gui = cv2.putText(image, f"{float(angle):.2f}", (200, 100), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2, cv2.LINE_AA)
        return overlayed_gui
