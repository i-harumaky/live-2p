import cv2
import os
import sys
import time
import numpy as np
from hand import Hand
from utils import rgb2bgr, Banner, zAdjuster

### personal settings ###
DOMINANT_HAND = "Left" # or "Right"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT= 720
BANNER_BACKGROUND= (230,230,230)
colors = [(13,18,23), (243,199,161), (29,132,200), (175,37,57)] # RGB
########################


os.chdir(os.path.dirname(__file__))
WINDOW_NAME = "Live 2P"

# カメラの用意
video_capture = cv2.VideoCapture(0)
if video_capture.isOpened():
    print('Video Capture available')
else:
    print('Unable to access the camera')
    sys.exit(1)

video_capture.set(3, CAMERA_WIDTH)
video_capture.set(4, CAMERA_HEIGHT)


handmod = Hand(detection_confidence=0.8, tracking_confidence=0.7)
prev_time, current_time = 0, 0

# Colors to BGR
colors = [rgb2bgr(c) for c in colors]
color = None # default: eraser

is_dominant_hand_left = DOMINANT_HAND == "Left"

banner = Banner(CAMERA_HEIGHT, colors, BANNER_BACKGROUND)
canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), np.uint8)
pix, piy = 0, 0 # Previous dominant hand's Index x/y
piix, piiy = 0, 0 # Previous nondominant hand's Index x/y
while True:
    # Read the image
    success, image = video_capture.read()
    image = cv2.flip(image, 1)

    # Parse the image
    handmod.parseHands(image)
    if handmod.hand_detected:
        handmod.draw(image)
        landmarks = handmod.getAllPosition(image)
        is_multi = (len(handmod.multi_handedness) == 2)
        hand_label = handmod.multi_handedness[0].classification[0].label
        is_dominant_hand = (hand_label == DOMINANT_HAND)

        if not is_multi and is_dominant_hand:
            ix, iy, iz = landmarks[8][0], landmarks[8][1], landmarks[8][2]
            finger_bins = handmod.getFingerBins(landmarks, hand_label=="Left")
            # drawing mode
            if finger_bins == [0,1,0,0,0]:
                if pix == 0 and piy == 0:
                    pix, piy = ix, iy
                
                cv2.line(canvas, (pix,piy), (ix,iy), color, zAdjuster(iz, 100))
                print(landmarks[8][2])
                pix, piy = ix, iy

            # selection mode
            else:
                pix, piy = 0, 0 # reset drawing init point
                cv2.rectangle(image, (ix-20,iy-20), (ix+20, iy+20), (255,255,255), thickness=-1)
                if ix < 180:
                    if iy < 180:
                        color = colors[0]
                        banner.select(0)
                    elif 180 < iy < 360:
                        color = colors[1]
                        banner.select(1)
                    elif 360 < iy < 540:
                        color = colors[2]
                        banner.select(2)
                    else: # ~720
                        color = colors[3]
                        banner.select(3)
        
        # sigle and nondominant
        # => eraser mode
        elif not is_multi:
            iix, iiy, iiz = landmarks[8][0], landmarks[8][1], landmarks[8][2]
            if piix == 0 and piiy == 0:
                piix, piiy = iix, iiy
            cv2.circle(image, (iix,iiy), zAdjuster(iiz, 500), (255,255,255), 3, lineType=cv2.LINE_AA)
            cv2.line(canvas, (piix,piiy), (iix,iiy), None, thickness=zAdjuster(iiz, 550))
            piix, piiy = iix, iiy

        # multi
        else:
            pix, piy, piix, piiy = 0, 0, 0, 0


    # Merge the image with canvas     
    image_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, image_inverse = cv2.threshold(image_gray, 1, 255, cv2.THRESH_BINARY_INV)
    image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, image_inverse)
    image = cv2.bitwise_or(image, canvas)

    
    image[0:720, 0:banner.width] = banner.image
    current_time = time.time()
    fps = 1/(current_time - prev_time)
    cv2.putText(image, str(int(fps)), (CAMERA_WIDTH-80, CAMERA_HEIGHT-40), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    cv2.imshow(WINDOW_NAME, image)
    prev_time = current_time
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # qで終了
        break


cv2.destroyAllWindows()