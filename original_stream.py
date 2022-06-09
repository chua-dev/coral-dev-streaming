import cv2
import numpy as np
from decouple import config


cap = cv2.VideoCapture(f"rtsp://{config('CCTV_USERNAME')}:{config('CCTV_PASSWORD')}@192.168.1.5:554/Stream/Channels/101")

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()