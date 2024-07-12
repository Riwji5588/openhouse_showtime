import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

# Load the image
main_image = cv2.imread('shit.jpg')
block_size = (200, 200)

# Resize the main image to be a multiple of the block size
main_image = cv2.resize(main_image, (block_size[0] * 5, block_size[1]))

# Extract blocks from the main image
blocks = [main_image[:, i*block_size[0]:(i+1)*block_size[0]] for i in range(5)]

# Initialize the camera and set dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Create reset button
resetButtonPos = (1150, 650)
resetButtonSize = (100, 50)

# Cooldown variables
cooldown_time = 60  # 60 seconds cooldown
last_reset_time = time.time()

class DragRect():
    def __init__(self, posCenter, size=[200, 200], img=None):
        self.posCenter = posCenter
        self.size = size
        self.img = img
        self.dragging = False
        self.initialPos = [posCenter[0], posCenter[1]]  # Store initial position

    def update(self, cursor):
        if self.dragging:
            self.posCenter = cursor

    def reset(self):
        self.posCenter = [self.initialPos[0], self.initialPos[1]]

rectList = []
for x in range(5):
    rect = DragRect([x * 250 + 150, 150], img=blocks[x])
    rectList.append(rect)

selectedRect = None
prevCursor = None

while True:
    current_time = time.time()
    elapsed_time = current_time - last_reset_time
    cooldown_remaining = max(0, cooldown_time - elapsed_time)

    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera. Ensure the camera is connected and the correct index is used.")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        lmList = hands[0]['lmList']
        cursor = lmList[8][:2]  # index finger tip landmark
        l, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)  # distance between index and middle finger tips

        # Check if the reset button is pressed with three fingers and cooldown is over
        if detector.fingersUp(hands[0]) == [1, 1, 1, 0, 0]:  # Thumb, index, and middle fingers are up
            thumbIndexDist, _, img = detector.findDistance(lmList[4][:2], lmList[8][:2], img)
            indexMiddleDist, _, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
            if thumbIndexDist < 50 and indexMiddleDist < 50:
                if resetButtonPos[0] < cursor[0] < resetButtonPos[0] + resetButtonSize[0] and \
                   resetButtonPos[1] < cursor[1] < resetButtonPos[1] + resetButtonSize[1]:
                    if cooldown_remaining <= 0:
                        for rect in rectList:
                            rect.reset()
                        last_reset_time = current_time

        # Dragging rectangles
        if l < 70:
            if selectedRect is None:
                for rect in rectList:
                    cx, cy = rect.posCenter
                    w, h = rect.size

                    if cx - w // 2 < cursor[0] < cx + w // 2 and \
                            cy - h // 2 < cursor[1] < cy + h // 2:
                        selectedRect = rect
                        rect.dragging = True
                        break
            if selectedRect:
                # Smooth movement and prevent block from going out of bounds
                if prevCursor is not None:
                    cursor = [(prevCursor[0] + cursor[0]) // 2, (prevCursor[1] + cursor[1]) // 2]
                cursor[0] = np.clip(cursor[0], selectedRect.size[0] // 2, 1280 - selectedRect.size[0] // 2)
                cursor[1] = np.clip(cursor[1], selectedRect.size[1] // 2, 720 - selectedRect.size[1] // 2)
                selectedRect.update(cursor)
                prevCursor = cursor
        else:
            if selectedRect:
                selectedRect.dragging = False
                selectedRect = None
                prevCursor = None
    else:
        prevCursor = None

    # Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        imgNew[cy - h // 2:cy + h // 2, cx - w // 2:cx + w // 2] = rect.img

    # Draw reset button
    cv2.rectangle(imgNew, resetButtonPos, (resetButtonPos[0] + resetButtonSize[0], resetButtonPos[1] + resetButtonSize[1]), (0, 255, 0), cv2.FILLED)
    cv2.putText(imgNew, 'Reset', (resetButtonPos[0] + 10, resetButtonPos[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw cooldown timer
    cv2.putText(imgNew, f"Cooldown: {int(cooldown_remaining)}s", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
