### IMPORTS ###


import os
import sys
import csv
import cv2
import glob
import datetime
import argparse
import numpy as np

from retinaface import RetinaFace
import face_model


### FACE DETECTOR MODEL ###


gpuid = 1
detector = RetinaFace('/model-r50/R50', 0, gpuid, 'net3')


### FEATURE EMBEDDING MODEL ###


parser = argparse.ArgumentParser(description='face model test')
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/model-r34-amf/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=1, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)


### VIDEO SET UP ###


cap = cv2.VideoCapture()
cap.open('/data/input.mp4')

thresh = 0.8
scales = [1024, 1980]

width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size    = (width,height)

fps = 30


### DIMENSIONAL SET UP ###


(ret, im) = cap.read()
if not ret:
    print('Could not load video frame 0. Exiting.')
    exit()
    
im_shape = im.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
im_scale = float(target_size) / float(im_size_min)
if np.round(im_scale * im_size_max) > max_size:
    im_scale = float(max_size) / float(im_size_max)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


### BEGIN INGESTING VIDEO ###


frame_counter = 0
frame_total = cap.get(7)
frames = []

rows = []

while True:

    (ret, im) = cap.read()

    # save to disk
    if not ret: 
    
        # save output csv
        with open('/data/output.csv', 'wb') as out_csv:
            wr = csv.writer(out_csv, quoting=csv.QUOTE_ALL)
            wr.writerows(rows)

#         # save output video
#         filename_video = '/data/output.avi'
#         out = cv2.VideoWriter(filename_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, size, True)
#         for i in range(len(frames)):
#             out.write(frames[i])
#         out.release()        

        break

    frame_counter = frame_counter + 1
    
    # run detector
    faces, landmarks = detector.detect(im, thresh, scales=[im_scale], do_flip=False)
    
    # process each face
    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            
            # resize to 112x112
            crop = im[box[1]:box[3], box[0]:box[2], :]
            crop112 = cv2.resize(crop, (112, 112))
    
            # tranpose
            crop112_t = np.transpose(crop112, (2,0,1))
            
            # extract 512 features
            features = model.get_feature(crop112_t)

            # save features to disk
            info_list = [[frame_counter, box[0], box[1], box[2], box[3]], features]
            row = []
            for sublist in info_list:
                for item in sublist:
                    row.append(item)
            rows.append(row)
            
#             # draw bbox rectangle
#             cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
#     
#     frames.append(im)

    if frame_counter % 100 == 0:
        print(str(frame_counter) + '/' + str(int(frame_total)))
        
cap.release()
