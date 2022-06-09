from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image 
from PIL import Image
from imutils.video import WebcamVideoStream

image = Image.open('facephoto1.jpeg')
#detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
detect_faces = FaceDetection(model_type=FaceDetectionModel.FULL)

'''
faces = detect_faces(image)
if not len(faces):
    print('no faces detected :(')
else:
    print(faces[0][0])
    print(faces[0][1])
    print(faces[0][2])
    render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
    render_to_image(render_data, image).show()
'''

import cv2
import numpy as np
from decouple import config

camera_id = 0
#camera_id = f"rtsp://{config('CCTV_USERNAME')}:{config('CCTV_PASSWORD')}@192.168.1.5:554/Stream/Channels/101"
cap = WebcamVideoStream(src=camera_id).start()

while(True):
    frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detect_faces(image)
    render_data = detections_to_render_data(faces, bounds_color=Colors.GREEN)
    print(render_data)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()