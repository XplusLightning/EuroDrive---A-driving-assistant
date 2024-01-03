import cv2
import numpy as np
from GeneralCVFunctions import GeneralFunctions
import pyautogui

GenFun = GeneralFunctions()
old_left_lane, old_right_lane = [0, 0, 0, 0], [0, 0, 0, 0]
live = True

image = cv2.imread('TestImages\Straight_Clear_Dawn.jpg')
image = cv2.imread('TestImages\Test1.png')
image = cv2.imread('TestImages\Test2.png')
image = cv2.imread('TestImages\Test3.png')
video = cv2.VideoCapture('TestVideos\RelativelyStraight_Clear_Dawn.mkv')
video = cv2.VideoCapture('TestVideos\Everything_Clear_Day.mkv')
video = cv2.VideoCapture('TestVideos\Straight_Rain_Overcast.mkv')
video = cv2.VideoCapture('TestVideos\Turns_Rain_Dusk.mkv')
video = cv2.VideoCapture('TestVideos\Turn_Clear_Dawn.mkv')
# video = cv2.VideoCapture(2)


while True:
    
    playing, image = video.read()
    # if not playing:
    #     break 
    
    if live:
        image = GenFun.screenshot()
    
    image_copy = np.copy(image)
    scaling_percent = 50
    mask_vertices = [(1300, 750), (500, 750), (700, 450), (1100, 450)]
    mask_vertices = [(1300, 750), (700, 750), (900, 450), (1100, 450)]
    
    scaled_copy = GenFun.resize(image_copy, scaling_percent)
    mask = GenFun.polygon_mask(image_copy, mask_vertices)
    scaled_mask = GenFun.resize(mask, 50)
    greyscaled_mask = GenFun.greyscale(scaled_mask)
    blur = GenFun.blur(greyscaled_mask)
    edges = GenFun.edge_detection(blur)
    # edges2 = GenFun.edge_detection(GenFun.blur(scaled_mask))
    second_mask = GenFun.polygon_mask(edges, GenFun.scaled_array(GenFun.adjusted_array(mask_vertices, 5), scaling_percent))
    left_lane, right_lane = GenFun.approximate_lane_lines(second_mask)
    
    if left_lane == [0, 0, 0, 0]:
        left_lane_image = GenFun.draw_lines(second_mask, np.array([old_left_lane]))
    else:
        left_lane_image = GenFun.draw_lines(second_mask, np.array([left_lane]))
        old_left_lane = left_lane.copy()
        
    if right_lane == [0, 0 ,0 ,0]:
        right_lane_image = GenFun.draw_lines(second_mask, np.array([old_right_lane]))
    else:
        right_lane_image = GenFun.draw_lines(second_mask, np.array([right_lane]))
        old_right_lane = right_lane.copy()
        
    lane_image = cv2.addWeighted(left_lane_image, 0.8, right_lane_image, 1, 1)
    overlayed_lanes = cv2.addWeighted(scaled_copy, 0.8, lane_image, 1, 1)
    # cv2.imshow("Image", scaled_copy)
    # cv2.imshow("Mask", scaled_mask)
    # cv2.imshow("Greyscaled Mask", greyscaled_mask)
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Edges2", edges2)
    cv2.imshow("Second Mask", second_mask)
    # cv2.imshow("Left Lane", left_lane_image)
    # cv2.imshow("Right Lane", right_lane_image)
    # cv2.imshow("Lanes", lane_image)
    cv2.imshow("Overlayed Lanes", overlayed_lanes)
    
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break
    
video.release()
cv2.destroyAllWindows()