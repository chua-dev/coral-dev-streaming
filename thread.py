import cv2
import queue
import time
import threading

q=queue.Queue()

from decouple import config
stream_source = f"rtsp://{config('CCTV_USERNAME')}:{config('CCTV_PASSWORD')}@192.168.1.5:554/Stream/Channels/101"

def Receive():
  print("start Receive")
  cap = cv2.VideoCapture(stream_source)
  ret, frame = cap.read()
  #q.put(frame)
  while ret:
      ret, frame = cap.read()
      q.put(frame)


def Display():
  print("Start Displaying")
  while True:
      if q.empty() != True:
        frame = q.get()
        cv2.imshow("the video", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
if __name__=='__main__':
  p1=threading.Thread(target=Receive)
  p2 = threading.Thread(target=Display)
  p1.start()
  p2.start()