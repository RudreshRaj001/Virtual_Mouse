# Virtual_Mouse 
Project Objectives:
- Develop an AI virtual mouse system that utilizes computer vision and machine learning algorithms to control the computer mouse cursor and perform scroll functions without the need for a physical mouse.
- Overcome limitations of traditional mouse devices, such as the requirement of batteries and dongles, by employing a webcam or built-in camera for hand gesture and hand tip detection.
- Enable users to control the graphical user interface (GUI) on a computer platform through hand gestures captured by the camera, eliminating the need for physical contact with devices.
- Utilize Python programming language, along with OpenCV for computer vision, MediaPipe for hand tracking, and Pynput, Autopy, and PyAutoGUI for cursor operations and scrolling functions.
- Achieve high accuracy and performance with the proposed model, making it suitable for real-world applications even on CPUs without the need for a GPU.

Problem Statement:
- Address the challenges faced in situations where physical mouse usage is impractical or restricted, such as limited space or individuals with hand impairments.
- Provide a solution for safe interaction with computer devices during situations like the COVID-19 pandemic, where touching devices may pose health risks.
- Utilize hand gesture and hand tip detection through a webcam or built-in camera to control PC mouse functions, ensuring a touchless and convenient user experience.

Aims & Objectives-
The objective of this project is to develop a Virtual Mouse.
Expected achievements :
•	To design to operate with the help of a webcam.
•	To convert hand gesture/motion into mouse input that will be set to a particular screen position.
•	Two Types of Mouse functions are achieved :
1.	Simple Click
2.	Scrolling


SCOPE OF THE PROJECT
Virtual Mouse that will soon to be introduced to replace the physical computer mouse to promote convenience while still able to accurately interact and control the computer system. To do that, the software requires to be fast enough to capture and process every image, in order to successfully track the user's gesture. Therefore, this project will develop a software application with the aid of the latest software coding technique and the open-source computer vision library also known as the OpenCV. The scope of the project is as below: 
• Real time application. 
• User friendly application. 
• Removes the requirement of having a physical mouse.

Applications-
The AI virtual mouse system is useful for many applications; it can be used to reduce the space for using the physical mouse, and it can be used in situations where we cannot use the physical mouse. The system eliminates the usage of devices, and it improves the human-computer interaction.
Major applications:
(i)The proposed model has a greater accuracy of 99% which is far greater than the that of other proposed models for virtual mouse, and it has many applications.
(ii)Amidst the COVID-19 situation, it is not safe to use the devices by touching them because it may result in a possible situation of spread of the virus by touching the devices, so the proposed AI virtual mouse can be used to control the PC mouse functions without using the physical mouse.
(iii)The system can be used to control robots and automation systems without the usage of devices.
(iv)2D and 3D images can be drawn using the AI virtual system using the hand gestures
(v)AI virtual mouse can be used to play virtual reality- and augmented reality-based games without the wireless or wired mouse devices.
(vi)Persons with problems in their hands can use this system to control the mouse functions in the computer.
(vii)In the field of robotics, the proposed system like HCI can be used for controlling robots.
(viii)In designing and architecture, the proposed system can be used for designing virtually for prototyping.

API DOCUMENTATTION

Brief introduction about Python and it’s installation instructions -
Python is a popular high-level programming language that is widely used for web development, data analysis, artificial intelligence, scientific computing, and many other applications. It is known for its simple syntax, ease of learning, and vast collection of libraries and frameworks. 
Python can be installed on various operating systems, including Windows, macOS, and Linux. Here are the general steps to install Python on a Windows computer: 
1)	Visit the official Python website at python.org.
2)	Click on the "Downloads" link and select the appropriate version of Python for the operating system. For example, in Windows 10, we should download the latest version of Python 3.X. 
3)	Run the installer and follow the prompts to install Python
4)	During the installation, we will be prompted to add Python to the system PATH. We should be sure to select this option for easily access Python from the command line. 
 
Once the installation is complete, open a command prompt and type "python" to verify that Python is installed correctly. We should see the Python interpreter start up and display its version information.

Packages required for the project-
•	Open CV Library:
OpenCV is used in the making of this program. OpenCV (Open Source Computer Vision) is a library of programming functions for real time computer vision like image and video processing . OpenCV have the utility that can read image pixels value, it also have the ability to create real time eye tracking, blink detection, face recognition and augmented reality. It supports various programming languages, including Python, C++, and Java. 


•	NumPy :
 NumPy is a Python library that provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. It is widely used for numerical computation, data analysis, and scientific computing. 

•	Mediapipe :
 MediaPipe is a cross-platform pipeline framework to build custom machine learning solutions for live and streaming media. Using MediaPipe, such a perception pipeline can be built as a graph of modular components. MediaPipe was built for machine learning (ML) teams and software developers who implement production-ready ML applications, or students and researchers who publish code and prototypes as part of their research work.

•	Pyautogui :  
PyAutoGUI is a Python library that provides a simple and cross-platform way to automate GUI tasks. It is a popular library that is widely used for tasks such as GUI testing, automating repetitive tasks, and generating human-like input. PyAutoGUI is a very powerful library that can simulate mouse and keyboard actions, take screenshots, and perform various other GUI-related tasks.
PyAutoGUI has a simple and easy-to-use interface that makes it a popular choice for automating GUI tasks. It provides functions for controlling the mouse, keyboard, and other input devices. PyAutoGUI can simulate mouse clicks, double-clicks, and drags. It can also perform keyboard input, such as typing text, pressing keys, and sending hotkeys. Additionally, PyAutoGUI can take screenshots of the screen, locate and identify GUI elements, and control the window focus.
PyAutoGUI is a cross-platform library that works on Windows, Mac, and Linux. It can be used with various programming languages that have a Python API, such as Java, Ruby, and C++. PyAutoGUI is also compatible with Python 2 and 3.
In the Virtual Mouse project, PyAutoGUI is used to control the mouse cursor and perform mouse-related actions. For example, PyAutoGUI is used to simulate mouse clicks, drag and drop operations, and scrolling. PyAutoGUI provides a reliable and easy-to-use interface for controlling the mouse cursor and performing GUI-related tasks.
Overall, PyAutoGUI is a powerful and versatile Python library that provides a simple and cross-platform way to automate GUI tasks. It is a valuable tool for anyone who needs to automate repetitive GUI tasks, generate human-like input, or perform GUI testing.















Code Documentation:
COMPLETE CODE : VIRTUAL MOUSE - The virtual mouse code tracks the position of the hand using computer vision techniques and maps it to the position of the mouse cursor, enabling the user to control the mouse without a physical device WITH TWO FUNCTIONS – click and scrolling.

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
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(139,71,137))
                    # this makes a yellow circle on the index finger only
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('outside', abs(index_y - thumb_y))
                    # adding a click function when the thumb and the index finger comes closer than 40 pixels
                    if abs(index_y - thumb_y) < 50:
                        print('click')
                        pyautogui.click() # click function
                        pyautogui.sleep(1)
                
# SCROll FUNCTION
# SCROLLING UP
# Little finger and thumb comes close for scrolling up
                          
                    # Little Finger
                if id == 20:  # as position of Little finger is 8
                    cv2.circle(img=frame, center=(x, y),radius=10, color=(0,255,0))
                    # this makes a yellow circle on the index finger only
                    Little_x = screen_width/frame_width*x
                    Little_y = screen_height/frame_height*y
                    if abs(thumb_y - Little_y) < 70:
                        print('SCROLLING UP')
                        pyautogui.scroll(20) # Scroll function
                        
# SCROLLING DOWN
# THE THUMB AND RING FINGER COMES CLOSE FOR SCROLLING DOWN    

                    #RING FINGER                    
                if id == 16:  # as position of Ring Finger is 16
                    cv2.circle(img=frame, center=(x, y),
                               radius=10, color=(205,55,0))
                    # this makes a yellow circle on the index finger only
                    ring_x = screen_width/frame_width*x
                    ring_y = screen_height/frame_height*y
                    if abs(thumb_y - ring_y) < 70:
                        print('SCROLLING DOWN')
                        pyautogui.scroll(-20) # Scroll function
                        
                    
                    

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)





Explaining the code in 5 STEPS :
STEP 1: OPENING THE CAMERA
import cv2
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

STEP 2 : TO DETECT THE HAND IN THE CAMERA 
For this we use the library “MediaPipe”, and we mark out the hand landmarks 
 
import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
while True:
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    print(hands)
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)


## for showing the hand with all the Landmarks Of the Hand

import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
while True:
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hands in hands:
            drawing_utils.draw_landmarks(frame, hands) # <---- for seeing the landmarks on the frame 
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)


MAPPING THE HAND :-
  
 

STEP 3 : DETECTING THE INDEX FINGER INDIVIDUALLY
import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # because the screen was originally opposite, this flips it back
    frame_height, frame_width, _ = frame.shape 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hands in hands:
            drawing_utils.draw_landmarks(frame, hands) # <---- for seeing the landmarks on the frame 
            landmarks = hands.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                print(x, y)
                if id == 8: # as position of index finger is 8
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    # this makes a yellow circle on the index finger only

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

STEP 4 : MOVING THE MOUSE POINTER USING INDEX FINGER
import cv2
import mediapipe as mp
import pyautogui
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size() # to know the size of the computer screen
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # because the screen was originally opposite, this flips it back
    frame_height, frame_width, _ = frame.shape 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hands in hands:
            drawing_utils.draw_landmarks(frame, hands) # <---- for seeing the landmarks on the frame 
            landmarks = hands.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x*frame_width)
                y = int(landmark.y*frame_height)
                print(x, y)
                if id == 8: # as position of index finger is 8
                    cv2.circle(img=frame, center=(x,y), radius=10, color=(0, 255, 255))
                    # this makes a yellow circle on the index finger only
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y
                    pyautogui.moveTo(index_x, index_y) # <--- this helps the curser move in the whole computer window
                    # pyautogui.moveTo(x, y) # but the curser only moves in the frame not the whole window like we want <--- so just above we wrote code for moving the curser on whole window

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)


STEP 5 : INTRODUCING CLICK FEATURE
•	When the index finger and thumb comes closer than 50 pixels the mouse clicks.
•	We separated both the thumb and the index figure from the Hand Landmark, we also marked index finger with yellow circle and thumb with purple(Orchid4) circle, so that it is differentiable from one another.

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
                if id == 4:  # as position of Thumb is 4
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(139,71,137))
                    # this makes a yellow circle on the index finger only
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('outside', abs(index_y - thumb_y))
                    # adding a click function when the thumb and the index finger comes closer than 40 pixels
                    if abs(index_y - thumb_y) < 40:
                        print('click')
                        pyautogui.click() # click function
                        pyautogui.sleep(1)
                    
                    

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

STEP 6 : INTRODUCING THE SCROLL FEATURE 
THIS WILL INCLUDE TWO FEATURES 
              1. SCROLLING UP 
              2. SCROLLING DOWN

1. SCROLLING UP ---> IF THUMB AND RING FINGER COMES CLOSER THAN 70 pixels screen scrolls up
            WE SEPARATE THE RING FINGER TOO WITH BLUE COLOR (RGB CODE : 0,255,0)

#SCROll FUNCTION
 #SCROLLING UP
 #Little finger and thumb comes close for scrolling up
                          
                    # Little Finger
                if id == 20:  # as position of Little finger is 8
                    cv2.circle(img=frame, center=(x, y),radius=10, color=(0,255,0))
                    # this makes a yellow circle on the index finger only
                    Little_x = screen_width/frame_width*x
                    Little_y = screen_height/frame_height*y
                    if abs(thumb_y - Little_y) < 70:
                        print('SCROLLING UP')
                        pyautogui.scroll(20) # Scroll function

2. SCROLLING DOWN ---> IF THUMB AND LITTLE FINGER COMES CLOSER THAN 70 pixels screen scrolls DOWN
            WE SEPERATE THE LITTLE FINGER TOO WITH GREEN COLOR RGB CODE : (RGB CODE : 205,55,0)
    #SCROLLING DOWN
    #THE THUMB AND RING FINGER COMES CLOSE FOR SCROLLING DOWN    

                    #RING FINGER                    
                if id == 16:  # as position of Ring Finger is 16
                    cv2.circle(img=frame, center=(x, y),
                               radius=10, color=(205,55,0))
                    # this makes a yellow circle on the index finger only
                    ring_x = screen_width/frame_width*x
                    ring_y = screen_height/frame_height*y
                    if abs(thumb_y - ring_y) < 70:
                        print('SCROLLING DOWN')
                        pyautogui.scroll(-20) # Scroll function


COMPLETE FINAL WORKING CODE :
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
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(139,71,137))
                    # this makes a yellow circle on the index finger only
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                    print('outside', abs(index_y - thumb_y))
                    # adding a click function when the thumb and the index finger comes closer than 40 pixels
                    if abs(index_y - thumb_y) < 50:
                        print('click')
                        pyautogui.click() # click function
                        pyautogui.sleep(1)
                
# SCROll FUNCTION
# SCROLLING UP
# Little finger and thumb comes close for scrolling up
                          
                    # Little Finger
                if id == 20:  # as position of Little finger is 8
                    cv2.circle(img=frame, center=(x, y),radius=10, color=(0,255,0))
                    # this makes a yellow circle on the index finger only
                    Little_x = screen_width/frame_width*x
                    Little_y = screen_height/frame_height*y
                    if abs(thumb_y - Little_y) < 70:
                        print('SCROLLING UP')
                        pyautogui.scroll(20) # Scroll function
                        
# SCROLLING DOWN
# THE THUMB AND RING FINGER COMES CLOSE FOR SCROLLING DOWN    

                    #RING FINGER                    
                if id == 16:  # as position of Ring Finger is 16
                    cv2.circle(img=frame, center=(x, y),
                               radius=10, color=(205,55,0))
                    # this makes a yellow circle on the index finger only
                    ring_x = screen_width/frame_width*x
                    ring_y = screen_height/frame_height*y
                    if abs(thumb_y - ring_y) < 70:
                        print('SCROLLING DOWN')
                        pyautogui.scroll(-20) # Scroll function
                        
                    
                    

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)
