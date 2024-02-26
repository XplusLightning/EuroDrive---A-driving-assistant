import cv2
import numpy as np
from GeneralCVFunctions import GeneralFunctions
from LaneControl import LaneControler


general = GeneralFunctions()
controler = LaneControler()
scaling_percent = 40
old_left_lane, old_right_lane = [0, 0, 0, 0], [0, 0, 0, 0]
mask_vertices = [(1300, 750), (700, 750), (925, 550), (1075, 550)]
video = cv2.VideoCapture("")
live = True
steering = False
k=0

while True:
    if live:
        image = general.screenshot()
    else:
        playing, image = video.read()
    image_copy = np.copy(image)
    
    mask = general.polygon_mask(image_copy, mask_vertices)
    greyscaled_mask = general.greyscale(mask)
    blur = general.blur(greyscaled_mask)
    edges = general.edge_detection(blur)
    second_mask = general.polygon_mask(edges, general.adjusted_array(mask_vertices, 5))
    left_lane, right_lane, straight_lines = general.approximate_lane_lines(second_mask)
    
    if left_lane == [0, 0, 0, 0]:
        left_lane_image = general.draw_lines(second_mask, np.array([old_left_lane]))
    else:
        left_lane_image = general.draw_lines(second_mask, np.array([left_lane]))
        old_left_lane = left_lane.copy()
        
    if right_lane == [0, 0, 0,0]:
        right_lane_image = general.draw_lines(second_mask, np.array([old_right_lane]))
    else:
        right_lane_image = general.draw_lines(second_mask, np.array([right_lane]))
        old_right_lane = right_lane.copy()
    
    lane_image = cv2.addWeighted(left_lane_image, 0.8, right_lane_image, 1, 1)
    overlayed_lanes = cv2.addWeighted(cv2.cvtColor(image_copy, cv2.COLOR_BGR2BGRA), 0.8, lane_image, 1, 1)
    
    angle = controler.lane_control(left_lane, right_lane, steering)
    lane_gui = general.lane_gui(overlayed_lanes, angle)
    
    
    cv2.imshow("Original", general.resize(image, scaling_percent))
    cv2.imshow("First Mask", general.resize(mask, scaling_percent))
    cv2.imshow("Straight Lanes", general.resize(straight_lines, scaling_percent))
    cv2.imshow("Overlayed Lanes", general.resize(lane_gui, scaling_percent))
    
    if cv2.waitKey(100) & 0xFF == ord('b'):
        break
    
video.release()
cv2.destroyAllWindows()