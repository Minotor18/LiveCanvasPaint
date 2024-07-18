import cv2
import mediapipe as mp
import time
import numpy as np
# Video capture ayarları
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Mediapipe el tespiti
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# FPS hesaplaması için zamanlayıcı
pTime = 0

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Renk
colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 0, 0)]
# Parmak uçlarının landmark indeksleri
finger_tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []
    tipID = [4,8,12,16,20]
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if len(lmList) > 20:
                        fingers = []
                        if lmList[tipID[0]][1] > lmList[tipID[0] - 1][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                        for id in range(1,5):
                            if lmList[tipID[id]][2] < lmList[tipID[id]-2][2]:
                                fingers.append(1)
                            else:
                                fingers.append(0)
                        if fingers[0] == 1:
                            cv2.circle(canvas, (lmList[tipID[0]][1], lmList[tipID[0]][2]), 50, colors[4], cv2.FILLED)
                        if fingers[1] == 1:
                            cv2.circle(canvas, (lmList[tipID[1]][1], lmList[tipID[1]][2]), 10,  colors[1], cv2.FILLED)
                        if fingers[2] == 1:
                            cv2.circle(canvas, (lmList[tipID[2]][1], lmList[tipID[2]][2]), 10,  colors[2], cv2.FILLED)
                        if fingers[3] == 1:
                            cv2.circle(canvas, (lmList[tipID[3]][1], lmList[tipID[3]][2]), 10,  colors[3], cv2.FILLED)
                        if fingers[4] == 1:
                            cv2.circle(canvas, (lmList[tipID[4]][1], lmList[tipID[4]][2]), 10,  colors[0], cv2.FILLED)
                        print(fingers)



    img = cv2.addWeighted(img, 0.7, canvas, 3, 0)
    # FPS hesaplaması
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # FPS metnini ekrana yazdırma
    cv2.putText(img, f"FPS: {fps:.2f}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



