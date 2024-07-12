import cv2
import numpy as np
import random
import time
from cvzone.HandTrackingModule import HandDetector
import cvzone

# Initialize camera and settings
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Define color constants
colorR = (255, 0, 255)  # Magenta color for rectangles

# Class to represent draggable rectangles
class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter  # Center position of the rectangle
        self.size = size            # Size of the rectangle

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if cursor (index finger tip) is inside the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

    def intersects(self, other):
        cx1, cy1 = self.posCenter
        w1, h1 = self.size
        cx2, cy2 = other.posCenter
        w2, h2 = other.size

        # Check if two rectangles intersect
        return (abs(cx1 - cx2) * 2 < (w1 + w2)) and (abs(cy1 - cy2) * 2 < (h1 + h2))

# Create a list of initial rectangles
rectList = [DragRect([random.randint(100, 1180), random.randint(100, 620)]) for _ in range(5)]

# Initialize game variables
score = 0
startTime = time.time()
duration = 60  # Duration of the game in seconds

# Main loop
while True:
    # Read frame from camera
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera. Make sure the camera is connected and the correct index is used.")
        break
    
    img = cv2.flip(img, 1)  # Flip the image horizontally for natural movement

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    # Update rectangle positions based on hand cursor (index finger tip)
    if hands:
        lmList = hands[0]['lmList']  # Landmark list of the first hand detected
        l, _, _ = detector.findDistance(lmList[8], lmList[12], img)  # Distance between thumb and index finger

        if l < 70:
            cursor = lmList[8]  # Index finger tip landmark
            # Update positions of all rectangles
            for rect in rectList:
                rect.update(cursor[:2])

    # Check for intersections between rectangles and update score
    for i in range(len(rectList)):
        for j in range(i + 1, len(rectList)):
            if rectList[i].intersects(rectList[j]):
                score += 1
                rectList.pop(j)  # Remove the intersecting rectangle
                break

    # Reset rectangles if only one remains
    if len(rectList) == 1:
        rectList = [DragRect([random.randint(100, 1180), random.randint(100, 620)]) for _ in range(5)]

    # Draw transparent rectangles on a new image
    imgNew = np.zeros_like(img, np.uint8)  # Create a blank image with the same dimensions as the input image
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)  # Draw rounded corners on the rectangle

    # Blend the new image with the original image to create transparency effect
    alpha = 0.5  # Transparency factor
    out = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)

    # Calculate remaining time and display score and timer
    elapsedTime = time.time() - startTime
    remainingTime = int(duration - elapsedTime)
    if remainingTime <= 0:
        break  # Exit the loop when time is up

    cv2.putText(out, f'Score: {score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(out, f'Time: {remainingTime}', (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the final output image
    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Break the loop if 'q' is pressed

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
