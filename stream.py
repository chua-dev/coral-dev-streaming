import cv2
import sys
import argparse

def main():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
    '--camera_id', help='Camera ID to be steram', default=0
  )

  args = parser.parse_args()

  camera_id = args.camera_id
  vid = cv2.VideoCapture(camera_id)

  while True:
    success, image = vid.read()

    # If source cannot be open / wrong source
    if not success:
        sys.exit(
            'ERROR: Unable to read from webcam. Please verify your webcam settings.'
        )

    cv2.imshow('Live Capture', image)
      
    if cv2.waitKey(1) == ord('q'):
      break



if __name__ == '__main__':
  main()