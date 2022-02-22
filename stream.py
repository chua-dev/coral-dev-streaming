import cv2
import sys
import argparse
import time

def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
    '--camera_id', help='Video source to be stream, if default webcam no need to specify', default=0,
  )

  parser.add_argument(
    '--fps_threshold', help='FPS threshold seperate color yellow or green', default=10, type=int
  )

  args = parser.parse_args()

  camera_id = args.camera_id
  fps_threshold = args.fps_threshold
  vid = cv2.VideoCapture(camera_id)

  prev_frame_time = 0
  new_frame_time = 0

  while True:
    start_time=time.time()
    success, image = vid.read()

    # If source cannot be open / wrong source
    if not success:
        sys.exit(
            'STREAM ERROR: Either one of below happen \n1) Unable to read from video source, please verify settings \n2) Video finish playing no more frame to display'
        )

    new_frame_time = time.time()
 
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    fps_int = int(fps)
    fps = str(fps_int)

    # Put Text Arguments
    fps_sentence = f'FPS: {fps}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_color = (80, 220, 100) if fps_int > fps_threshold else (34, 201, 255)
    thickness = 6
    font_scale = 2
    cv2.putText(image, fps_sentence, (50, 80), font, font_scale, image_color, thickness, cv2.LINE_4)

    cv2.imshow('Live Capture', image)
    if cv2.waitKey(1) == ord('q'):
      break

if __name__ == '__main__':
  main()