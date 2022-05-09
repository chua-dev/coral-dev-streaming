#from tkinter import Frame
import cv2
import sys
import argparse
import time
import imutils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import tensorflow as tf
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pdb

# Helper
from helper_function.model_function import generate_anchors
from helper_function.model_function import decode_boxes
from helper_function.model_function import non_max_suppression

# Initialize Constant Var
input_mean = 127.5
input_std = 127.5
NMS_UNSPECIFIED_OVERLAP_TYPE = 0
NMS_JACQUARD = 1
NMS_MODIFIED_JACCARD = 2
NMS_INTERSECTION_OVER_UNION = 3

NMS_DEFAULT = 0
NMS_WEIGHTED = 1

NUM_KEYPOINTS_PER_BOX = 6
NUM_COORDS_PER_KEYPOINT = 2

# Initiate Flask App
from flask import Flask, render_template, Response
app = Flask(__name__)


from decouple import config
stream_source = f"rtsp://{config('CCTV_USERNAME')}:{config('CCTV_PASSWORD')}@192.168.1.5:554/Stream/Channels/101"

# Setup TF Lite Model
interpreter = tf.lite.Interpreter(model_path="model/face_detection_full_range.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
output_details = interpreter.get_output_details()

# Setting Var after getting input detail
FACE_ANCHOR_OPTIONS = {
  "num_layers": 1,
  "min_scale": 0.1484375,
  "max_scale": 0.75,
  "input_size_height": input_height,
  "input_size_width": input_width,
  "anchor_offset_x": 0.5,
  "anchor_offset_y": 0.5,
  "strides": [4],
  "aspect_ratios": [1.0],
  "fixed_anchor_size": True,
  "interpolated_scale_aspect_ratio": 0.0,
  "reduce_boxes_in_lowest_layer": False,
}

FACE_ANCHORS = generate_anchors(FACE_ANCHOR_OPTIONS)

FACE_DECODE_OPTIONS = {
  "num_classes": 10,
  "num_boxes": 2304,
  "num_coords": 16,
  "keypoint_coord_offset": 4,
  "num_keypoints": 6,
  "num_values_per_keypoint": 2,
  "box_coord_offset": 0,
  "x_scale": 192.0,
  "y_scale": 192.0,
  "h_scale": 192.0,
  "w_scale": 192.0,
  "apply_exponential_on_box_size": False,
  "reverse_output_order": True,
  "sigmoid_score": True,
  "score_clipping_threshold": 100.0,
  "flip_vertically": False,
  "min_score_threshold": 0.6,
}

FACE_NMS_OPTIONS = {
  "max_num_detections": -1,
  "min_score_threshold": 0.1,
  "min_suppression_threshold": 0.3,
  "overlap_type": NMS_INTERSECTION_OVER_UNION,
  "algorithm": NMS_WEIGHTED,
  "num_boxes": 2304,
  "num_coords": 16,
  "keypoint_coord_offset": 4,
  "num_keypoints": 6,
  "num_values_per_keypoint": 2,
  "box_coord_offset": 0,
}

parser = argparse.ArgumentParser(
  formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
  #'-cam', '--camera_id', help='Video source to be stream, if default webcam no need to specify', default=stream_source,
  '-cam', '--camera_id', help='Video source to be stream, if default webcam no need to specify', default=0,
)

parser.add_argument(
  '-fps', '--fps_threshold', help='FPS threshold seperate color yellow(low) or green(satisfy)', default=10, type=int
)

parser.add_argument(
  '-f', '--flask_app', help='Display using flask app or imshow, default=True', action='store_false', default = True
)

parser.add_argument(
  '-s', '--size', help="frame size rescale", default=800
)

args = parser.parse_args()

camera_id = args.camera_id
fps_threshold = args.fps_threshold
use_flaskapp = args.flask_app 
rescale_size = args.size
print(use_flaskapp)

#vid = cv2.VideoCapture(camera_id)
vid = WebcamVideoStream(src=camera_id).start()


def main():
  prev_frame_time = 0
  new_frame_time = 0

  while True:
    start_time=time.time()
    image = vid.read()

    if image.any():
      # FPS Thingy
      new_frame_time = time.time()
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time
      fps_int = int(fps)
      fps = str(fps_int)

      image = imutils.resize(image, width=rescale_size)

      # Image Processing For Detection Model
      processing_image = cv2.resize(image, (input_width, input_height))
      processing_image = np.expand_dims(processing_image, axis=0)
      processing_image = (np.float32(processing_image) - input_mean) / input_std
      processed_image = np.float32(processing_image)

      # Invoke Detection & Get Result
      interpreter.set_tensor(input_details[0]['index'], processed_image)
      interpreter.invoke()

      # Getting Output & Result
      rel_coordinate = interpreter.get_tensor(output_details[0]['index'])[0]
      scores = interpreter.get_tensor(output_details[1]['index'])[0]
      indexed_scores = np.flip(np.argsort(scores.flatten()))

      pdb.set_trace()

      # Put Text Arguments
      fps_sentence = f'FPS: {fps}'
      font = cv2.FONT_HERSHEY_SIMPLEX
      image_color = (80, 220, 100) if fps_int > fps_threshold else (34, 201, 255)
      thickness = 5
      font_scale = 1.2
      cv2.putText(image, fps_sentence, (50, 80), font, font_scale, image_color, thickness, cv2.LINE_4)
      
      # Stream on browser app
      ret, buffer = cv2.imencode('.jpg', image)
      image = buffer.tobytes()
      yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

      if cv2.waitKey(1) == ord('q'):
        break

# Imshow Streaming
def main2():
  prev_frame_time = 0
  new_frame_time = 0

  while True:
    start_time=time.time()
    #success, image = vid.read()
    image = vid.read()

    if image:
      new_frame_time = time.time()
  
      fps = 1/(new_frame_time-prev_frame_time)
      prev_frame_time = new_frame_time

      fps_int = int(fps)
      fps = str(fps_int)

      image = imutils.resize(image, width=rescale_size)
      height = image.shape[0]

      # Image Processing For Detection Model
      processing_image = cv2.resize(image, (input_width, input_height))
      processing_image = np.expand_dims(processing_image, axis=0)
      processing_image = (np.float32(processing_image) - input_mean) / input_std
      processed_image = np.float32(processing_image)

      # Invoke Detection & Get Result
      interpreter.set_tensor(input_details[0]['index'], processed_image)
      interpreter.invoke()

      # Getting Output & Result
      rel_coordinate = interpreter.get_tensor(output_details[0]['index'])[0]
      scores = interpreter.get_tensor(output_details[1]['index'])[0]
      indexed_scores = np.flip(np.argsort(scores.flatten()))

      boxes = decode_boxes(FACE_DECODE_OPTIONS, rel_coordinate.flatten(), scores, FACE_ANCHORS)
      clean_boxes = non_max_suppression(FACE_NMS_OPTIONS, scores, boxes, NMS_WEIGHTED)

      for box in clean_boxes:
        min_x = box["rect"]["min_x"] * input_width * rescale_size / 192
        min_y = box["rect"]["min_y"] * input_height * height / 192
        max_x = box["rect"]["max_x"] * input_width * rescale_size / 192
        max_y = box["rect"]["max_y"] * input_height * height / 192
        start_point = (int(min_x),int(min_y))
        end_point = (int(max_x),int(max_y))
        image = cv2.rectangle(image, start_point, end_point, (0,255,0), 1)
      #pdb.set_trace()

      # Put Text Arguments
      fps_sentence = f'FPS: {fps}'
      font = cv2.FONT_HERSHEY_SIMPLEX
      image_color = (80, 220, 100) if fps_int > fps_threshold else (34, 201, 255)
      thickness = 5
      font_scale = 1.2
      cv2.putText(image, fps_sentence, (50, 80), font, font_scale, image_color, thickness, cv2.LINE_4)
      cv2.imshow('Live Capture', image)
      #print(image.shape)

      if cv2.waitKey(1) == ord('q'):
        break


@app.route('/')
def video_feed():
  return Response(main(), mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__ == '__main__':
  if use_flaskapp: 
    app.run(host='0.0.0.0', port=5000, threaded=True) 
  else: 
    main2()