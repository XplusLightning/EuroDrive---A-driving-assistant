import cv2
import numpy as np
import pyautogui
from time import sleep 
from GeneralCVFunctions import GeneralFunctions
from LaneControl import LaneControler


general = GeneralFunctions()
controler = LaneControler()
scaling_percent = 50
old_left_lane, old_right_lane = [0, 0, 0, 0], [0, 0, 0, 0]
mask_vertices = [(1300, 750), (500, 750), (700, 450), (1100, 450)]
# mask_vertices = [(1300, 750), (900, 750), (1000, 550), (1100, 550)]
mask_vertices = [(1300, 750), (700, 750), (925, 550), (1075, 550)]
live = False
steering = True

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
video = cv2.VideoCapture('TestVideos\Italy_Turns_Clear.mkv')


while True:
    
    playing, image = video.read()
    # if not playing:
    #     break 
    
    if live:
        image = general.screenshot()
        sleep(0.1)
    else:
        sleep(0.03)    
    image_copy = np.copy(image)
    
    # scaled_copy = general.resize(image_copy, scaling_percent)
    mask = general.polygon_mask(image_copy, mask_vertices)
    # scaled_mask = general.resize(mask, 50)
    greyscaled_mask = general.greyscale(mask)
    blur = general.blur(greyscaled_mask)
    edges = general.edge_detection(blur)
    # edges2 = general.edge_detection(general.blur(scaled_mask))
    second_mask = general.polygon_mask(edges, general.adjusted_array(mask_vertices, 5))
    left_lane, right_lane = general.approximate_lane_lines(second_mask)
    
    if left_lane == [0, 0, 0, 0]:
        left_lane_image = general.draw_lines(second_mask, np.array([old_left_lane]))
    else:
        left_lane_image = general.draw_lines(second_mask, np.array([left_lane]))
        old_left_lane = left_lane.copy()
        
    if right_lane == [0, 0 ,0 ,0]:
        right_lane_image = general.draw_lines(second_mask, np.array([old_right_lane]))
    else:
        right_lane_image = general.draw_lines(second_mask, np.array([right_lane]))
        old_right_lane = right_lane.copy()
    lane_image = cv2.addWeighted(left_lane_image, 0.8, right_lane_image, 1, 1)
    overlayed_lanes = cv2.addWeighted(cv2.cvtColor(image_copy, cv2.COLOR_BGR2BGRA), 0.8, lane_image, 1, 1)
    
    angle = controler.lane_control(left_lane, right_lane, steering)
    lane_gui = general.lane_gui(overlayed_lanes, angle)
    overlayed_gui = cv2.addWeighted(overlayed_lanes, 0.8, lane_gui, 1, 1)
    overlayed_gui = cv2.putText(overlayed_gui, f"{float(angle):.2f}", (100, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2, cv2.LINE_AA)
    
    # cv2.imshow("Image", scaled_copy)
    # cv2.imshow("Mask", scaled_mask)
    cv2.imshow("Greyscaled Mask", general.resize(greyscaled_mask))
    # cv2.imshow("Blur", blur)
    # cv2.imshow("Edges", edges)
    # cv2.imshow("Edges2", edges2)
    cv2.imshow("Second Mask", general.resize(second_mask))
    # cv2.imshow("Left Lane", left_lane_image)
    # cv2.imshow("Right Lane", right_lane_image)
    # cv2.imshow("Lanes", lane_image)
    cv2.imshow("Overlayed Lanes", general.resize(overlayed_gui, scaling_percent))
    
    if cv2.waitKey(10) & 0xFF == ord('b'):
        break
    
video.release()
cv2.destroyAllWindows()