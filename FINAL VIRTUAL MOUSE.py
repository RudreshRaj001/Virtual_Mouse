import cv2
import mediapipe as mp
import pyautogui
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
# to know the size of the computer screen
screen_width, screen_height = pyautogui.size()
index_y = 0
while True:
    _, frame = cap.read()
    # because the screen was originally opposite, this flips it back
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            # <---- for seeing the landmarks on the frame
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                # print(x, y)

                # INDEX FINGER
                if id == 8:  # as position of index finger is 8
                    cv2.circle(img=frame, center=(x, y),
                               radius=10, color=(0, 255, 255))
                    # this makes a yellow circle on the index finger only
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y
                    # <--- this helps the curser move in the whole computer window
                    pyautogui.moveTo(index_x, index_y)
                    # pyautogui.moveTo(x, y) # but the curser only moves in the frame not the whole window like we want <--- so just above we wrote code for moving the curser on whole window

                    # MAKING THE THUMB SEPERATE TOO SO THAT WE CAN MAKE A GESTURE AND CREATE A CLICKING OPTION

# CLICK FUNCTION
# Thumb and Index finger comes close for click

                    # THUMB
                if id == 4:  # as position of Thumb is 4
                    cv2.circle(img=frame, center=(x, y),
                               radius=10, color=(139, 71, 137))
                    # this makes a yellow circle on the index finger only
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('outside', abs(index_y - thumb_y))
                    # adding a click function when the thumb and the index finger comes closer than 40 pixels
                    if abs(index_y - thumb_y) < 50:
                        print('click')
                        pyautogui.click()  # click function
                        pyautogui.sleep(1)

# SCROll FUNCTION
# SCROLLING UP
# Little finger and thumb comes close for scrolling up

                    # Little Finger
                if id == 20:  # as position of Little finger is 8
                    cv2.circle(img=frame, center=(x, y),radius=10, color=(0, 255, 0))
                    # this makes a yellow circle on the index finger only
                    Little_x = screen_width/frame_width*x
                    Little_y = screen_height/frame_height*y
                    if abs(thumb_y - Little_y) < 70:
                        print('SCROLLING UP')
                        pyautogui.scroll(100)  # Scroll function

# SCROLLING DOWN
# THE THUMB AND RING FINGER COMES CLOSE FOR SCROLLING DOWN

                    # RING FINGER
                if id == 16:  # as position of Ring Finger is 16
                    cv2.circle(img=frame, center=(x, y),
                               radius=10, color=(205, 55, 0))
                    # this makes a yellow circle on the index finger only
                    ring_x = screen_width/frame_width*x
                    ring_y = screen_height/frame_height*y
                    if abs(thumb_y - ring_y) < 70:
                        print('SCROLLING DOWN')
                        pyautogui.scroll(-100)  # Scroll function

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)
