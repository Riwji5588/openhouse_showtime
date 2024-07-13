import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import random
import os
import pandas as pd

# Paths and settings
main_image_path = 'future_entity.png'
sword_image_path = 'sword.png'
image_save_path = './img'
excel_file_path = './excel/game_scores.xlsx'
block_size = (200, 200)

# Load the main image
main_image = cv2.imread(main_image_path)
if main_image is None:
    print(f"Error: Unable to load image at {main_image_path}")
    exit(1)

# Load the sword image
sword_image = cv2.imread(sword_image_path, cv2.IMREAD_UNCHANGED)
if sword_image is None:
    print(f"Error: Unable to load image at {sword_image_path}")
    exit(1)

# Resize the main image to be a multiple of the block size

main_image = cv2.resize(main_image, block_size)



# Initialize the camera and set dimensions
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Create Start button
startButtonPos = (550, 300)
startButtonSize = (200, 100)
start_hold_time = 2  # Hold for 2 seconds to start
start_hold_start = None
game_started = False
game_ended = False

# Cooldown variables
cooldown_time = 60  # 60 seconds cooldown
last_reset_time = time.time()
sword_image_resized = cv2.resize(sword_image, (100, 100))
# Create reset button
resetButtonPos = (1150, 20)
resetButtonSize = (100, 50)

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
        while True:
            self.posCenter = [random.randint(0, 1280 - self.size[0]), random.randint(0, 720 - self.size[1])]
            if not self.check_overlap():  # Ensure no overlap with other blocks
                break
        self.img = self.initialImg  # Reset to the initial image

    def check_overlap(self):
        for rect in rectList:
            if rect != self:
                cx, cy = rect.posCenter
                w, h = rect.size
                if not (self.posCenter[0] + self.size[0] <= cx or self.posCenter[0] >= cx + w or
                        self.posCenter[1] + self.size[1] <= cy or self.posCenter[1] >= cy + h):
                    return True
        return False

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
rectList_Last = []
for x in range(2):
    rect = DragRect([x * 250 + 150, 150], size=block_size, img=main_image[:, x * block_size[0]:(x + 1) * block_size[0]])
    rectList.append(rect)

selectedRect = None
prevCursor = None
score = 0

def check_hand_touching_sword(cursor, sword_posCenter, sword_size):
    sx, sy = sword_posCenter
    sw, sh = sword_size
    return True

def notify_player():
    # Display a message on the screen to notify the player
    notification_start_time = time.time()
    while time.time() - notification_start_time < 3:  # Show message for 3 seconds
        success, img = cap.read()
        if not success:
            print("Failed to read frame from camera. Ensure the camera is connected and the correct index is used.")
            break
        img = cv2.flip(img, 1)
        cv2.putText(img, "Get Ready! Capturing Image...", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def capture_image():
    # Capture an image and save it to the specified directory
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{image_save_path}/player_image_{timestamp}.png"
        cv2.imwrite(filename, img)
        return filename
    else:
        print("Failed to capture image.")
        return None

def save_score_to_excel(image_filename, score):
    # Save the image filename and score to an Excel file
    data = {
        'Image Filename': [image_filename],
        'Score': [score]
    }
    df = pd.DataFrame(data)
    if os.path.exists(excel_file_path):
        df_existing = pd.read_excel(excel_file_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_excel(excel_file_path, index=False)

def futureMaki_HakiMode():
    rectList.clear()

while True:
    current_time = time.time()
    elapsed_time = current_time - last_reset_time
    cooldown_remaining = max(0, cooldown_time - elapsed_time)

    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera. Ensure the camera is connected and the correct index is used.")
        break

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if not game_started:
        rectList.clear()
        # Draw Start button
        game_ended = False
        cv2.rectangle(img, startButtonPos, (startButtonPos[0] + startButtonSize[0], startButtonPos[1] + startButtonSize[1]), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, 'Start Game', (startButtonPos[0] + 10, startButtonPos[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if hands:
            lmList = hands[0]['lmList']
            cursor = lmList[8][:2]  # index finger tip landmark
            cx = startButtonPos[0]
            cy = startButtonPos[1]
            w = 200
            h = 100
            if startButtonPos[0] < cursor[0] < startButtonPos[0] + startButtonSize[0] and startButtonPos[1] < cursor[1] < startButtonPos[1] + startButtonSize[1]:
                if start_hold_start is None:
                    start_hold_start = time.time()
                elif time.time() - start_hold_start > start_hold_time:
                    notify_player()
                    image_filename = capture_image()
                    if image_filename:
                        game_started = True
                        for x in range(5):
                            rect = DragRect([x * 250 + 150, 150], size=block_size, img=main_image)
                            rectList.append(rect)
                        last_reset_time = time.time()
                        start_hold_start = None

            if cx < cursor[0] < cx + w and cy < cursor[1] < cy + h:
                img_last = rect.img.copy()
                rect.hide()  # Hide the block
                if not np.array_equal(img_last, rect.img):
                    if score % 5 == 0:
                        for reset_rect in rectList:
                            reset_rect.reset()
                            reset_rect.show()
            else:
                start_hold_start = None

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if game_started and not game_ended:
        if hands:
            lmList = hands[0]['lmList']
            cursor = lmList[8][:2]  # index finger tip landmark
            l, info, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)  # distance between index and middle finger tips
            
            sword_posCenter = cursor
            scaled_sword_size = (sword_image_resized.shape[1], sword_image_resized.shape[0])

            if sword_posCenter[0] >= 0 and sword_posCenter[0] + scaled_sword_size[0] <= 1280 \
                    and sword_posCenter[1] >= 0 and sword_posCenter[1] + scaled_sword_size[1] <= 720:
                alpha_sliced = sword_image_resized[:, :, 3] / 255.0
                alpha_sliced = alpha_sliced[:, :, np.newaxis]
                img[sword_posCenter[1]:sword_posCenter[1] + scaled_sword_size[1],
                    sword_posCenter[0]:sword_posCenter[0] + scaled_sword_size[0], :3] = \
                    alpha_sliced * sword_image_resized[:, :, :3] + \
                    (1 - alpha_sliced) * img[sword_posCenter[1]:sword_posCenter[1] + scaled_sword_size[1],
                                            sword_posCenter[0]:sword_posCenter[0] + scaled_sword_size[0], :3]

            if detector.fingersUp(hands[0]) == [1, 1, 1, 0, 0]:  # Thumb, index, and middle fingers are up
                thumbIndexDist, _, img = detector.findDistance(lmList[4][:2], lmList[8][:2], img)
                indexMiddleDist, _, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                if thumbIndexDist < 50 and indexMiddleDist < 50:
                    if cooldown_remaining <= 0:
                        for rect in rectList:
                            rect.reset()
                        last_reset_time = current_time
            # Check if hand is touching the image
            i = 0
            for rect  in rectList:
                cx, cy = rect.posCenter
                w, h = rect.size
                if cx < cursor[0] < cx + w and cy < cursor[1] < cy + h:
                    img_last = rect.img.copy()
                    num_lastarray = len(rectList)
                    rectList.pop(i)
                    num_array = len(rectList)
                    # rect.hide()  # Hide the block
                    if num_array != num_lastarray:
                        score += 1
                      # Generate a new block
                        if score % 5 == 0:
                            for x in range(5):
                                rect = DragRect([x * 250 + 150, 150], size=block_size, img=main_image)
                                rectList.append(rect)
                i += 1

        if cooldown_remaining <= 0:
            game_ended = True
            save_score_to_excel(image_filename, score)

    # Draw Transparency
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        if rect.img is not None:
            img[cy:cy + h, cx:cx + w] = rect.img

    # Draw score
    cv2.putText(img, f"Score: {score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw cooldown timer in top right corner
    if not game_ended:
        cv2.putText(img, f"Cooldown: {int(cooldown_remaining)}s", (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(img, f"Final Score: {score}", (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(img, "Game Over", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        game_started = False
        score = 0

    # Overlay images with transparency
    out = img.copy()
    alpha = 0   # set effect of the image
    mask = img.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, img, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
