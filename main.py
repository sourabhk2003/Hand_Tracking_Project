import cv2
import mediapipe as mp
import numpy as np
import os
import math

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Camera settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 5)
width, height = 1280, 720
cap.set(3, width)
cap.set(4, height)

# Canvas for drawing
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Load header images
folderPath = "Hand Tracking Project/Header"
if not os.path.exists(folderPath):
    print(f"Error: Folder not found - {folderPath}")
    exit()

myList = os.listdir(folderPath)
print("Header files found:", myList)

overlayList = []
for imPath in myList:
    imagePath = os.path.join(folderPath, imPath)
    image = cv2.imread(imagePath)
    if image is None:
        print(f"Error: Unable to load image - {imagePath}")
        continue
    overlayList.append(image)

if not overlayList:
    print("Error: No valid header images found.")
    exit()

header = overlayList[0]
drawColor = (0, 0, 255)
thickness = 20
tipIds = [4, 8, 12, 16, 20]
xp, yp = 0, 0

# MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Flip the image and process
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Convert back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks and process gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = [[int(lm.x * width), int(lm.y * height)] for lm in hand_landmarks.landmark]

                if points:
                    x1, y1 = points[8]
                    x2, y2 = points[12]

                    fingers = [
                        1 if points[tipIds[0]][0] < points[tipIds[0] - 1][0] else 0,
                        *[
                            1 if points[tipIds[i]][1] < points[tipIds[i] - 2][1] else 0
                            for i in range(1, 5)
                        ],
                    ]

                    # Selection Mode
                    if fingers[1] and fingers[2] and all(fingers[i] == 0 for i in [0, 3, 4]):
                        xp, yp = x1, y1
                        if y1 < 125:
                            if 170 < x1 < 295:
                                header = overlayList[0]
                                drawColor = (0, 0, 255)
                            elif 436 < x1 < 561:
                                header = overlayList[1]
                                drawColor = (255, 0, 0)
                            elif 700 < x1 < 825:
                                header = overlayList[2]
                                drawColor = (0, 255, 0)
                            elif 980 < x1 < 1105:
                                header = overlayList[3]
                                drawColor = (0, 0, 0)
                        cv2.rectangle(image, (x1 - 10, y1 - 15), (x2 + 10, y2 + 23), drawColor, cv2.FILLED)

                    # Draw Mode
                    if fingers[1] and all(fingers[i] == 0 for i in [0, 2, 3, 4]):
                        cv2.circle(image, (x1, y1), thickness // 2, drawColor, cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        xp, yp = x1, y1

                    # Clear the canvas
                    if all(fingers[i] == 0 for i in range(5)):
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = 0, 0

        # Merge drawings with the original image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, imgInv)
        image = cv2.bitwise_or(image, imgCanvas)

        # Set the header
        image[0:125, 0:width] = header

        cv2.imshow("Virtual Painter", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
