import cv2
import numpy as np
from mss import mss 
from PIL import Image
from math import sin, cos, radians

class GeneralFunctions:
    def __init__(self):
        pass 
    
    
    def resize(self, image, scale_percent=50):
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
    
    
    def approximate_lane_lines(self, image):
        if image is None:
            raise Exception("Image is None")

        lines = cv2.HoughLinesP(image, 2, np.pi/180, 100, minLineLength=20, maxLineGap=200)
        if lines is None:
            lines = []
        left_lane, right_lane = self.lane_approximations(image, lines)
 
        return left_lane, right_lane
    
    def draw_lines(self, image, lines):
        blank = cv2.cvtColor(np.zeros_like(image), cv2.COLOR_GRAY2BGRA)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line 
                if x1 != x2:
                    if abs((y2-y1)/(x2-x1)) > 0.2:
                        blank = cv2.line(blank, (x1, y1), (x2, y2), (0, 255, 0, 1), 2)
                else:
                    pass
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
            else:
                pass
        
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
        y1 = image.shape[0] - 0
        y2 = int(y1 * 0.3)
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
    
    
    def lane_gui(self, image, angle, x_intersect=-1, steering_direction=""):
        x, y = 0, 0  
        if angle > 10:
            foreground = self.resize(cv2.cvtColor(cv2.imread("direction_images\\right.png"), cv2.COLOR_BGR2BGRA), 50)
        elif angle < -10:
            foreground = self.resize(cv2.cvtColor(cv2.imread("direction_images\left.png"), cv2.COLOR_BGR2BGRA), 50)
            
        else:
            foreground = self.resize(cv2.cvtColor(cv2.imread("direction_images\straight.png"), cv2.COLOR_BGR2BGRA), 50)
        rows, cols, channels = foreground.shape
        image[y:y+rows, x:x+cols] = foreground
        
        x1, y1 = 156, 120
        rad = 40
        point1 = (x1, y1)
        point2 = (int(rad*sin(radians(angle))+x1), y1-int(rad*cos(radians(angle))))
        image = cv2.arrowedLine(image, point1, point2, (0, 0, 255, 1), 2)
        overlayed_gui = cv2.putText(image, f"{float(angle):.2f}", (150, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)
        overlayed_gui = cv2.putText(image, f"{float(x_intersect):.2f}", (150, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)
        overlayed_gui = cv2.putText(image, steering_direction, (150, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1, cv2.LINE_AA)

        return overlayed_gui


    def mask_centre(self, vertices):
        x1, y1 = vertices[0]
        x2, y2 = vertices[1]
        
        return (int((x1+x2)/2), int((y1+y2)/2))
    
    
    def mean_line(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        blank = np.zeros_like(image)
        # print(image)
        const = int(image.shape[0] * 0.3)
        const2 = int(image.shape[0] * 0.8)
        x_coords = []
        y_coords = []
        for i, row in enumerate(image[const:const2]):
            row_mean = self.mean_row(row)
            if row_mean != None:
                x_coords.append(row_mean)
                y_coords.append(i+const)
        
        if len(x_coords) == 0:
            return blank, 1, 0

        slope, y_intercept = np.polyfit(x_coords, y_coords, 1)
        y1 = int(image.shape[0] * 0)
        y2 = int(image.shape[0])
        x1 = int((y1-y_intercept)/slope)
        x2 = int((y2-y_intercept)/slope)
        
        blank = cv2.line(blank, (x1, y1), (x2, y2), 100, 1)
        for i, x in enumerate(x_coords):
            blank = cv2.line(blank, (int(x), int(y_coords[i])), (int(x), int(y_coords[i])), 255, 1)
        
        return blank, slope, y_intercept
    
    
    def mean_row(self, array):
        total = 0
        if sum(array) == 0:
            return None
        
        for i, num in enumerate(array):
            total += i*num
        index = total/sum(array)
        
        return index 
    
    
    def barrier_check(self, array):
        pixel_range = 3
        image = np.zeros_like(array)
        # middle_column_index = int(array.shape[1]/2)
        middle_column_index = int(array.shape[1]/2)
        
        
        for row in range(0, array.shape[0], pixel_range):
            for column in range(middle_column_index, array.shape[1]):
                if 0 < column < array.shape[1] and 0 < row + pixel_range < array.shape[0]:
                    if any(array[row+i][column] == 255 for i in range(pixel_range)):
                        for k in range(pixel_range):
                                image[row+k][column] = 255
                        break
                    
        for row in range(0, array.shape[0], pixel_range):
            for column in range(middle_column_index, 0, -1):
                if row + pixel_range < array.shape[0]:
                    if any(array[row+i][column] == 255 for i in range(pixel_range)):
                        for k in range(pixel_range):
                                image[row+k][column] = 255
                        break
        
        return image
    
    
    def middle_of_barriers(self, array):
        image = np.zeros_like(array)
        angle = 180
        x_intersect = 128
        x_coords = []
        y_coords = []
        
        for row in range(0, array.shape[0]):
            if np.sum(array[row]==255) == 2:
                x_coords.append(np.mean(np.where(array[row]==255)))
                y_coords.append(row)
        
        if len(x_coords) == 0:
            return image, angle, x_intersect, image
        
        for i, x in enumerate(x_coords):
            image = cv2.line(image, (int(x), int(y_coords[i])), (int(x), int(y_coords[i])), 255, 1)
        
        filtered_barriers, total_coords = self.find_line(image)
        if total_coords is None:
            slope, intersect = np.polyfit(x_coords, y_coords, 1)
        else: 
            slope, intersect = np.polyfit((total_coords[0][0], total_coords[0][2]), (total_coords[0][1], total_coords[0][3]), 1)
            
        x1, y1, x2, y2 = self.make_coordinates(array, (slope, intersect))
        angle = np.degrees(np.arctan(1/slope)) * -1
        x_intersect = (image.shape[0] - intersect)/slope
        
        if x1-x2 != 0:
            image = cv2.line(image, (x1, y1), (x2, y2), 100, 1)
        
        
        return image, angle, x_intersect, filtered_barriers


    def find_line(self, image):
        image = cv2.Canny(image, 50, 150, apertureSize=7)
        lines = cv2.HoughLinesP(image, 1, np.pi/180, 30, minLineLength=10, maxLineGap=10)
        line_image = np.zeros_like(image)
        total = np.array([[0, 0, 0, 0]])
        
        if lines is not None:
            for i in lines:
                total += i
                x1, y1, x2, y2 = i.reshape(4)
                line_image = cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            # print(total)
            total = total/len(lines)
            
        return line_image, total if lines is not None else None



if __name__ == '__main__':
    a= GeneralFunctions()
    image = a.screenshot()
    image = a.resize(image, 20)
    image = a.greyscale(image)
    print(image.shape)
    barrier = a.barrier_check(image)
    middle = a.middle_of_barriers(barrier)
    cv2.imshow('barrier', barrier)
    cv2.imshow('middle', middle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    