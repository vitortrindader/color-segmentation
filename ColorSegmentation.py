# Necessary Imports

import cv2
import numpy as np

cap = cv2.VideoCapture(0) # Access your webcam
while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # In image classification is prefered to use HSV rather than RGB
    
    # Red Color
    low_red = np.array([161,155,84])
    high_red = np.array([179,255,255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame,frame, mask = red_mask)
    
    # Blue Color
    low_blue = np.array([94,80,2])
    high_blue = np.array([126,255,255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame,frame, mask = blue_mask)
    
    # Green Color
    
    low_green = np.array([25,52,72])
    high_green = np.array([102,255,255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame,frame, mask = green_mask)    
    
    # Every Color less white
    
    low = np.array([0,42,0])
    high = np.array([179,255,255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame,frame, mask = mask)
    
    # Showing
    
    cv2.imshow("Frame",frame)
    cv2.imshow("Less white", mask) # Choose the color that you want to do the segmation
    key = cv2.waitKey(1)
    if cv2.waitKey(10) & 0xFF == ord('q'): # Press 'q' to close the cam tab
        break
        
cap.release()
cv2.destroyAllWindows()