import face_model
import argparse
import cv2
import sys
import numpy as np

import os
import glob
import pickle

from PIL import Image

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='./models/model-r100-ii/model,0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from beginning')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)

real_fnames = glob.glob('../../real_70000_ffhq/*.png')

# print(len(real_fnames))

real_encodings = {}

for i, fname in enumerate(real_fnames[0:100]):

    print(i, fname)
    
    img = cv2.imread(fname)    
    inp = model.get_input(img)
    
    real_encodings[i] = {
        'fname_real_70000': fname.split('/')[-1],
        'enc': model.get_feature(inp),
    }


with open('real_encodings_arc.pkl', 'wb') as f:
    pickle.dump(real_encodings, f)

exit()
    
# img = cv2.imread('Tom_Hanks_54745.png')
# print(f1[0:10])
# gender, age = model.get_ga(img)
print(gender)
print(age)
sys.exit(0)
img = cv2.imread('/raid5data/dplearn/megaface/facescrubr/112x112/Tom_Hanks/Tom_Hanks_54733.png')
f2 = model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
