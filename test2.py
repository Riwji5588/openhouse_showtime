import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import random
import time

# Try different camera indices if 2 doesn't work (0, 1, etc.)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

    def intersects(self, other):
        cx1, cy1 = self.posCenter
        w1, h1 = self.size
        cx2, cy2 = other.posCenter
        w2, h2 = other.size

        return (abs(cx1 - cx2) * 2 < (w1 + w2)) and (abs(cy1 - cy2) * 2 < (h1 + h2))

rectList = []
for x in range(5):
    rectList.append(DragRect([random.randint(100, 1180), random.randint(100, 620)]))

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
            for rect in rectList:
                rect.update(cursor)
    
    # Check for intersections and increase score
    for i, rect1 in enumerate(rectList):
        for j, rect2 in enumerate(rectList):
            if i != j and rect1.intersects(rect2):
                score += 1
                rectList.pop(j)
                break
        else:
            continue
        break
    
    if len(rectList) == 1:
        rectList = []
        for x in range(5):
            rectList.append(DragRect([random.randint(100, 1180), random.randint(100, 620)]))

    ## Draw Transperency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

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
