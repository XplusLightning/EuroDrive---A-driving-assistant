import cv2
import numpy as np
from GeneralCVFunctions import GeneralFunctions
from LaneControl import LaneControler


general = GeneralFunctions()
controler = LaneControler()
scaling_percent = 40
old_left_lane, old_right_lane = [0, 0, 0, 0], [0, 0, 0, 0]
mask_vertices = [(1300, 750), (700, 750), (925, 550), (1075, 550)]
video = cv2.VideoCapture("TestVideos//2024-02-10 13-29-18.mkv")
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
    # cv2.imshow("Greyscaled Mask", general.resize(greyscaled_mask, scaling_percent))
    # cv2.imshow("Blur", general.resize(blur, scaling_percent))
    # cv2.imshow("Edges", general.resize(edges, scaling_percent))
    # cv2.imshow("Second Mask", general.resize(second_mask, scaling_percent))
    # cv2.imshow("Left Lane", general.resize(left_lane_image, scaling_percent))
    # cv2.imshow("Right Lane", general.resize(right_lane_image, scaling_percent))
    # cv2.imshow("Lanes", general.resize(lane_image, scaling_percent))
    cv2.imshow("Overlayed Lanes", general.resize(lane_gui, scaling_percent))
    
    if cv2.waitKey(100) & 0xFF == ord('b'):
        break
    # if cv2.waitKey(100) & 0xFF == ord('c'):
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Original_CV_{k}.png", image)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Hough_Lines_CV_{k}.png", cv2.cvtColor(straight_lines, cv2.COLOR_BGRA2BGR))
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\First Mask_CV_{k}.png", mask)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Greyscaled Mask_CV_{k}.png", greyscaled_mask)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Blur_CV_{k}.png", blur)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Edges_CV_{k}.png", edges)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Second Mask_CV_{k}.png", second_mask)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Left Lane_CV_{k}.png", left_lane_image)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Right Lane_CV_{k}.png", right_lane_image)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Lanes_CV_{k}.png", lane_image)
        # cv2.imwrite(f"OpenCV_Assistant\\BasicLaneFinding\\Images\\Overlayed Lanes_CV_{k}.png", lane_gui)

        k+=1
    
video.release()
cv2.destroyAllWindows()