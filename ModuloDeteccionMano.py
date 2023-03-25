import cv2
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # Funcion de mediapipe para dibujar puntos de referencia

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = self.hands.process(imgRGB)
        #print(resultado.multi_hand_landmarks)

        if resultado.multi_hand_landmarks:
            for handLms in resultado.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #Muestra los puntos
                                                                                # y conexiones
        return img


                #for id, lm in enumerate(handLms.landmark):  # coordenadas de cada punto de referencia
                    # print(id,lm)
                    #h, w, c = img.shape  # Obtiene el ancho, alto de los puntos de referencia de la imagen
                    #cx, cy = int(lm.x * w), int(lm.y * h)  # obtiene las cordenadas o ubicacion de los puntos en pixeles
                    #print(id, cx, cy)

                    # if id == 0:   #Remarca puntos de referencia
                    #cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
                    #articulares en la mano detectada



def main():

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__== "__main__":
    main()

