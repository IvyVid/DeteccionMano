import cv2
import mediapipe as mp
import time
# Verifica la velocidad de fotograma

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #Funcion de mediapipe para dibujar puntos de referencia

pTime = 0 #Tiempo anterior
cTime = 0 #Tiempo actual


while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultado = hands.process(imgRGB)
    #print(resultado.multi_hand_landmarks)

    if resultado.multi_hand_landmarks:
        for handLms in resultado.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): #coordenadas de cada punto de referencia
                #print(id,lm)
                h, w, c = img.shape  #Obtiene el ancho, alto de los puntos de referencia de la imagen
                cx, cy = int(lm.x*w), int(lm.y*h) #obtiene las cordenadas o ubicacion de los puntos en pixeles
                print(id, cx, cy)


                if id == 0:   #Remarca puntos de referencia
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #Muestra los puntos y conexiones articulares en la mano detectada


    cTime = time.time()  #Tiempo actual : extrae la hora en un cierto tiempo
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
