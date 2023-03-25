import cv2
import time
import numpy as np
import mediapipe as mp

#-----------------------
wCam, hCam = 500, 480
#Ancho y alto de la camara para definir la imagen
#------------------------

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()


    #------------Detecta las manos---------------

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imgRGB)
    # print(resultado.multi_hand_landmarks)

    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:

            handNo = 0
            lmList = []

            for id, lm in enumerate(handLms.landmark):  # coordenadas de cada punto de referencia
                # print(id,lm)
                h, w, c = img.shape  # Obtiene el ancho, alto de los puntos de referencia de la imagen
                cx, cy = int(lm.x * w), int(lm.y * h)  # obtiene las cordenadas o ubicacion de los puntos en pixeles
                #print(id, cx, cy)
                lmList.append([id, cx, cy])

                if len(lmList) != 0:

                    d = lmList[0][:]
                    #print (lmList[4], lmList[17])
                    print(d)


                    #x1, y1 = lmList[4][1], lmList[4][2]
                    #x2, y2 = lmList[17][1], lmList[17][2]

                    #cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    #cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)

                #print(lmList)

                #if id == 4:  # Remarca puntos de referencia
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms,
                                  mpHands.HAND_CONNECTIONS)  # Muestra los puntos y conexiones articulares en la mano detectada

    #--------------------------------------------------------------------------


    #cv2.imshow("Imagen", img)

    # Flip the image horizontally for a selfie-view display.
    #cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    #cv2.waitKey(1)
    #if cv2.waitKey(5) & 0xFF == 27:

    cv2.imshow("Imagen", cv2.flip(img, 1))
    cv2.waitKey(1)


