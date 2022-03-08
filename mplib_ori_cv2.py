from distutils.log import error
from tkinter import Frame
from unittest import result
import cv2
import sys
import argparse
import time
import mediapipe as mp
from sklearn import model_selection
import pdb
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#pdb.set_trace()

face_detection = mp_face_detection.FaceDetection(0.5)

# Initiate Flask App
from flask import Flask, render_template, Response
app = Flask(__name__)

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
  '-cam', '--camera_id', help='Video Source', default=0,
  #'-cam', '--camera_id', help='Video source', default="rtsp://admin:abc12345@192.168.1.5:554/Stream/Channels/101",
)

parser.add_argument(
  '-fps', '--fps_threshold', help='FPS threshold seperate color yellow(low) or green(satisfy)', default=10, type=int
)

parser.add_argument(
  '-f', '--flask_app', help='Display using flask app or imshow, default=True', action='store_false'
)

parser.add_argument(
  '-s', '--size', help="frame size rescale", default=800
)

args = parser.parse_args()

camera_id = args.camera_id
fps_threshold = args.fps_threshold
use_flaskapp = args.flask_app 
rescale_size = args.size
print(f'Using Flask App: {use_flaskapp}')
#vid = WebcamVideoStream(src=camera_id).start()
vid = cv2.VideoCapture(camera_id)

def main():
  prev_frame_time = 0
  new_frame_time = 0

  while True:
    try:
      start_time=time.time()
      success, image = vid.read()

      new_frame_time = time.time()
  
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time

      fps_int = int(fps)
      fps = str(fps_int)
      image = imutils.resize(image, width=rescale_size)

      # Put Text Arguments
      fps_sentence = f'FPS: {fps}'
      font = cv2.FONT_HERSHEY_SIMPLEX
      image_color = (80, 220, 100) if fps_int > fps_threshold else (34, 201, 255)
      thickness = 5
      font_scale = 1.2
      cv2.putText(image, fps_sentence, (50, 80), font, font_scale, image_color, thickness, cv2.LINE_4)

      # Mediapipe preprocess
      #image.flags.writable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      #image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)
      
      # Stream on browser app
      ret, buffer = cv2.imencode('.jpg', image)
      image = buffer.tobytes()
      yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

      if cv2.waitKey(1) == ord('q'):
        break
    except Exception as e:
      print(e)

# Imshow Streaming
def main2():
  prev_frame_time = 0
  new_frame_time = 0

  while True:
    try:
      start_time=time.time()
      success, image = vid.read()

      new_frame_time = time.time()
  
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time

      fps_int = int(fps)
      fps = str(fps_int)

      image = imutils.resize(image, width=rescale_size)

      # Put Text Arguments
      fps_sentence = f'FPS: {fps}'
      font = cv2.FONT_HERSHEY_SIMPLEX
      image_color = (80, 220, 100) if fps_int > fps_threshold else (34, 201, 255)
      thickness = 5
      font_scale = 1.2
      cv2.putText(image, fps_sentence, (50, 80), font, font_scale, image_color, thickness, cv2.LINE_4)

      # Mediapipe preprocess
      #image.flags.writable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      #image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)

      cv2.imshow('Live Capture', image)

      if cv2.waitKey(1) == ord('q'):
        break
    except Exception as e:
      print(e)


@app.route('/')
def video_feed():
  return Response(main(), mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__ == '__main__':
  if use_flaskapp:
    app.run(host='0.0.0.0', port=5000, threaded=True) 
  else: 
    main2()