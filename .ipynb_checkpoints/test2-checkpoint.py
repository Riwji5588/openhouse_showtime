import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random
import time

# Try different camera indices if 0 doesn't work (1, 2, etc.)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

class DragCircle():
    def __init__(self, posCenter, img, radius=100):
        self.posCenter = posCenter
        self.radius = radius
        self.img = cv2.resize(img, (radius*2, radius*2)) if img is not None else None

    def update(self, cursor):
        cx, cy = self.posCenter
        r = self.radius

        # If the index finger tip is in the circle region
        if (cx - cursor[0]) ** 2 + (cy - cursor[1]) ** 2 < r ** 2:
            # Ensure the circle stays within the bounds of the frame
            cx_new = np.clip(cursor[0], r, 1280 - r)
            cy_new = np.clip(cursor[1], r, 720 - r)
            self.posCenter = [cx_new, cy_new]

    def intersects(self, other):
        cx1, cy1 = self.posCenter
        r1 = self.radius
        cx2, cy2 = other.posCenter
        r2 = other.radius

        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        return distance < r1 + r2

circleImages = ["circle1.png", "circle2.png", "circle3.png", "circle4.png", "circle5.png"]
circleList = []

for imgPath in circleImages:
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image {imgPath}. Using fallback.")
    circleList.append(DragCircle([random.randint(100, 1180), random.randint(100, 620)], img))

score = 0
startTime = time.time()
duration = 60  # 60 seconds countdown timer

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera. Ensure the camera is connected and the correct index is used.")
        break
    
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']
        l, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
        if l < 70:
            cursor = lmList[8][:2]  # index finger tip landmark
            # call the update here
            for circle in circleList:
                circle.update(cursor)
    
    # Check for intersections and increase score
    for i, circle1 in enumerate(circleList):
        for j, circle2 in enumerate(circleList):
            if i != j and circle1.intersects(circle2):
                score += 1
                circleList.pop(j)
                break
        else:
            continue
        break
    
    if len(circleList) == 1:
        circleList = []
        for imgPath in circleImages:
            img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not load image {imgPath}. Using fallback.")
            circleList.append(DragCircle([random.randint(100, 1180), random.randint(100, 620)], img))

    ## Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for circle in circleList:
        cx, cy = circle.posCenter
        r = circle.radius
        if circle.img is not None:
            # Create circular mask
            mask = np.zeros((r*2, r*2, 4), dtype=np.uint8)
            cv2.circle(mask, (r, r), r, (255, 255, 255, 255), -1)
            img_circle = cv2.bitwise_and(circle.img, mask)
            
            # Ensure the overlay is within frame boundaries
            y1, y2 = max(0, cy-r), min(720, cy+r)
            x1, x2 = max(0, cx-r), min(1280, cx+r)
            y1o, y2o = max(0, r-(cy-y1)), min(r*2, r+(y2-cy))
            x1o, x2o = max(0, r-(cx-x1)), min(r*2, r+(x2-cx))
            
            try:
                imgNew[y1:y2, x1:x2] = cv2.addWeighted(imgNew[y1:y2, x1:x2], 0, img_circle[y1o:y2o, x1o:x2o], 1, 0)
            except Exception as e:
                print(f"Exception in blending: {e}")
        else:
            # Draw fallback circle if image is not available
            cv2.circle(imgNew, (cx, cy), r, colorR, cv2.FILLED)

    out = img.copy()
    alpha = 0.5
    mask = imgNew[:, :, 3] > 0
    for c in range(3):
        out[:, :, c] = np.where(mask, cv2.addWeighted(img[:, :, c], alpha, imgNew[:, :, c], 1 - alpha, 0), out[:, :, c])

    # Calculate remaining time
    elapsedTime = time.time() - startTime
    remainingTime = int(duration - elapsedTime)
    if remainingTime <= 0:
        break  # Exit the loop when time is up

    # Display score and timer
    cv2.putText(out, f'Score: {score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(out, f'Time: {remainingTime}', (1100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", out)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
