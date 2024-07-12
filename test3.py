import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import random

# Load the image
main_image = cv2.imread('shit.jpg')
block_size = (200, 200)

# Resize the main image to be a multiple of the block size
main_image = cv2.resize(main_image, (block_size[0] * 5, block_size[1]))

# Initialize the camera and set dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Create reset button
resetButtonPos = (1150, 20)
resetButtonSize = (100, 50)

# Cooldown variables
cooldown_time = 10000  # 60 seconds cooldown
last_reset_time = time.time()

class DragRect():
    def __init__(self, posCenter, size=[200, 200], img=None):
        self.posCenter = posCenter
        self.size = size
        self.img = img
        self.initialImg = img.copy()  # Store the initial image for reset
        self.dragging = False
        self.reset()

    def update(self, cursor):
        if self.dragging:
            self.posCenter = cursor

    def reset(self):
        self.posCenter = [random.randint(0, 1280 - self.size[0]), random.randint(0, 720 - self.size[1])]
        self.img = self.initialImg  # Reset to the initial image

    def hide(self):
        self.img = np.zeros_like(self.img)  # Hide by blanking out the image

    def show(self):
        if self.initialImg is not None:
            h, w, _ = self.initialImg.shape
            if h != self.size[1] or w != self.size[0]:
                self.img = cv2.resize(self.initialImg, (self.size[0], self.size[1]))
        else:
            self.img = main_image[:, self.posCenter[0]:self.posCenter[0] + self.size[0], self.posCenter[1]:self.posCenter[1] + self.size[1]]

rectList = []
for x in range(5):
    rect = DragRect([x * 250 + 150, 150], size=block_size, img=main_image[:, x * block_size[0]:(x + 1) * block_size[0]])
    rectList.append(rect)

selectedRect = None
prevCursor = None
score = 0

sword_image = cv2.imread('sword.jpg')

# Function to check if hand is touching the sword image
def check_hand_touching_sword(cursor, sword_posCenter, sword_size):
    sx, sy = sword_posCenter
    sw, sh = sword_size
    if sx < cursor[0] < sx + sw and sy < cursor[1] < sy + sh:
        return True
    return False


while True:
    current_time = time.time()
    elapsed_time = current_time - last_reset_time
    cooldown_remaining = max(0, cooldown_time - elapsed_time)
    Shape = 0
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
                if cooldown_remaining <= 0:
                    for rect in rectList:
                        rect.reset()
                    last_reset_time = current_time

        # Check if hand is touching the image
        for rect in rectList:
            cx, cy = rect.posCenter
            w, h = rect.size
            print(rect.img)
            if cx < cursor[0] < cx + w and cy < cursor[1] < cy + h:
                 if check_hand_touching_sword(cursor, rect.posCenter, rect.size):
            # Here you can manipulate the sword image or interact with it
            # For example, hide or modify it based on game logic
            
                    img_last = rect.img.copy()
                    rect.hide()  # Hide the block
                    # Compare img_last and current rect.img
                    if not np.array_equal(img_last, rect.img):
                        score += 1
                        print(score)
                        if score % 5 == 0:
                            for reset_rect in rectList:
                                reset_rect.reset()
                                reset_rect.show()
       

    # Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        imgNew[cy:cy + h, cx:cx + w] = rect.img

    # Draw reset button
    # cv2.rectangle(imgNew, resetButtonPos, (resetButtonPos[0] + resetButtonSize[0], resetButtonPos[1] + resetButtonSize[1]), (0, 255, 0), cv2.FILLED)
    # cv2.putText(imgNew, 'Reset', (resetButtonPos[0] + 10, resetButtonPos[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw score
    cv2.putText(imgNew, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw cooldown timer in top right corner
    cv2.putText(imgNew, f"Cooldown: {int(cooldown_remaining)}s", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    # print(cooldown_remaining)
    if cooldown_remaining <= 0:
        print("Cooldown time is over. Exiting the program.")
        break

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
