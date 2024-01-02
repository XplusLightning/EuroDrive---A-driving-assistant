import cv2
import numpy as np

class GeneralFunctions:
    def __init__(self):
        pass 
    
    def resize(self, image, scale_percent):
        if image is None:
            raise Exception("Image is None")
        
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        new_dimensions = (width, height)
        return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA) 
    
    def rectangle_mask(self, image, vertices):
        if image is None:
            raise Exception("Image is None")
        
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, vertices[0], vertices[1], 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image
    # (1300, 750), (1300, 300), (500, 750), (500, 300)
    
    def polygon_mask(self, image, vertices):
        if image is None:
            raise Exception("Image is None")
        
        vertices = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(image, image, mask=mask)
    
    
    def greyscale(self, image):
        if image is None:
            raise Exception("Image is None")
        
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    
    def blur(self, image, kernel_size=5):
        if image is None:
            raise Exception("Image is None")
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    
    def edge_detection(self, image, threshold_1=50, threshold_2=150):
        if image is None:
            raise Exception("Image is None")
        
        return cv2.Canny(image, threshold_1, threshold_2)
    
    
    def show(self, image, window_name="Image"):
        if image is None:
            raise Exception("Image is None")
        
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def straight_lines(self, image):
        if image is None:
            raise Exception("Image is None")

        lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, minLineLength=50, maxLineGap=500)
        if lines is None:
            lines = []
        approximate_lanes = self.lane_approximations(image, lines)
        if approximate_lanes is None:
            return []
        return approximate_lanes
    
    def draw_lines(self, image, lines):
        blank = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line 
                # slope, y_intercept = np.polyfit((x1, x2), (y1, y2), 1)
                if abs((y2-y1)/(x2-x1)) > 0.2:
                    cv2.line(blank, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
            return None
        left_fit = []
        right_fit = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
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
            left_line = np.array([0, 0, 0, 0])
            
        if len(right_fit) != 0:
            right_fit_average = np.average(right_fit, axis=0)
            right_line = self.make_coordinates(image, right_fit_average)
        else:
            right_line = np.array([0, 0, 0, 0])
        
        return np.array([left_line, right_line])
    
    
    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0] - 150
        y2 = int(y1 * 0.6)
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        return np.array([x1, y1, x2, y2])
                    
                    