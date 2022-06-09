import tensorflow as tf
import numpy as np

model_file = 'model/face_landmark.tflite'

interpreter = tf.lite.Interpreter(
    model_path=model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f'input_details {input_details}')
print(f'output_details {output_details}')

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32
print(f'is floating model: {floating_model}')

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

import cv2
import matplotlib.pyplot as plt
image = cv2.imread('facephoto3.jpeg')
#imgplot = plt.imshow(image[:,:,::-1])


image_cropped = image[:image.shape[0], :image.shape[0], ::-1] # crop to avoid letterboxing step
#imgplot = plt.imshow(image_cropped)

img = cv2.resize(image_cropped, (192, 192))[np.newaxis, :, :, :]
img = (np.float32(img) - 0.0) / 255.0  # normalization
#imgplot = plt.imshow(img.squeeze())
#plt.show()

interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

output_face_landmarks = interpreter.get_tensor(output_details[0]['index'])[0]
output_face_flag = interpreter.get_tensor(output_details[1]['index'])[0]

print(output_face_flag)
print(output_face_landmarks.squeeze()[0:9])
print(output_face_landmarks.squeeze().shape)

output_face_landmarks = tf.reshape(tensor=output_face_landmarks, shape=(468,3))
print(output_face_landmarks)

face_landmark_x = output_face_landmarks[:, 0:1]
face_landmark_y = output_face_landmarks[:, 1:2]

imgplot = plt.imshow(image_cropped)
imgplot = plt.plot(face_landmark_x/192.0*image_cropped.shape[0], (face_landmark_y/192.0)*image_cropped.shape[1], '*')
plt.show()


imgplot = plt.imshow(image[:,:,::-1])
imgplot = plt.plot(face_landmark_x/192.0*image_cropped.shape[0], (face_landmark_y/192.0)*image_cropped.shape[1], '*')
#plt.show()