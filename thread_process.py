from threading import Thread
import cv2
import cv2
import mediapipe as mp
import multiprocessing
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras_facenet import FaceNet
import datetime
from keras.models import load_model

cap = cv2.VideoCapture(0)
pTime = 0
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.45,1)
embedder = FaceNet()
#embedder = load_model('/model/facenet_keras_weights.h5')
tempSentRecord = {}

x_list = list()
y_list = list()
name_dict = dict()
#path = r'D:\Users\Admin\Desktop\clientFaces\clients3'
#path = "/Users/p2digital/Documents/theface"
i = 0
for folders in os.listdir(path):
    try:
        name_dict[i] = folders
        truePath = os.path.join(path,folders)
        for image in os.listdir(truePath):
            img = cv2.imread(os.path.join(truePath,image))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            imgEncode = embedder.extract(img, threshold=0.95)[0]['embedding']
            x_list.append(imgEncode)
            y_list.append(i)
        i += 1
    except:
        print('unbuildable folder')

# Setting IP camera Stream Link From Env
from decouple import config
stream_source = f"rtsp://{config('CCTV_USERNAME')}:{config('CCTV_PASSWORD')}@192.168.1.5:554/Stream/Channels/101"


class VideoStreamWidget(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self,record,event):
        # Display frames in main program
        while True:
            if self.status:
                self.frame = self.maintain_aspect_ratio_resize(self.frame, width=600)
                #cv2.imshow('IP Camera Video Streaming', self.frame)
                imgRGB = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                results = faceDetection.process(imgRGB)
                bbox_data = list()
                if results.detections:
                    for ids, detection in enumerate(results.detections):
                        #mpDraw.draw_detection(img, detection)
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, ic = img.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                                int(bboxC.width * iw), int(bboxC.height * ih)
                        
                        bbox_data.append(img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
                        #cv2.rectangle(img,bbox,(255,0,255),2)
                        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(255,0,255),2)
                    record["bbox_data"] = bbox_data    
                    event.set()
                ### CHANGE DISPLAY METHOD, OR STREAM ON WEB

            # Press Q on keyboard to stop recording
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.capture.release()
                cv2.destroyAllWindows()
                exit(1)

        # added from here
        return self.frame
        # to here

    # Resizes a image and maintains aspect ratio
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        # We are resizing height if width is none
        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the 0idth and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)

def recognize_face(encodeFace,x_list,y_list,name_dict,threshold):
    dist_list = list()
    for face in x_list:
        dist_list.append(np.linalg.norm(encodeFace-face))
    index1 = np.argmin(dist_list)
    #print(dist_list)
    #print(index1)
    if dist_list[index1] < threshold:
        index2 = y_list[index1]
        return name_dict[index2]
    else:
        pass

def face_recognition(record,event):
    while True:
        event.wait()
        img_list = record["bbox_data"]
        try:
            for img in img_list:
                encode = embedder.embeddings([img])
                name = recognize_face(encode,x_list,y_list,name_dict,0.83)
                #if name != None:
                if name in tempSentRecord:
                    time_diff = datetime.datetime.now() - tempSentRecord[name]
                    time_sec = time_diff.total_seconds()
                    time_min = time_sec/60
                    if time_min <= 1: #minit
                        print(f'{name} still within 1min cooldown period')
                        pass
                    else:
                        print(name)
                        tempSentRecord[name] = datetime.datetime.now()
                else:
                    print(name)
                    tempSentRecord[name] = datetime.datetime.now()
        except:
            pass
        event.clear()

# Actual Running Of Stream
if __name__ == '__main__':
    stream_link = 0
    video_stream_widget = VideoStreamWidget(stream_link)

    # creating processes
    manager = multiprocessing.Manager()
    events = multiprocessing.Event()

    records = manager.dict(bbox_data = '')
	
    p1 = multiprocessing.Process(target=video_stream_widget.show_frame, args=(records,events))
    p2 = multiprocessing.Process(target=face_recognition, args=(records,events))

    # starting process 1
    p1.start()
	# starting process 2
    p2.start()

	# wait until process 1 is finished
    p1.join()
	# wait until process 2 is finished
    p2.join()

	# both processes finished
    print("Done!")