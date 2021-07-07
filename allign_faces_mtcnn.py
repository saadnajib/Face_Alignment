from mtcnn import MTCNN
import cv2
from scipy.spatial import distance
import numpy as np
import math 
from PIL import Image
 

def alignment_procedure(img, left_eye, right_eye):
    
    #this function aligns given face in img based on left and right eye coordinates
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
 
    #-----------------------
    #find rotation direction
    
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
    
    #-----------------------
    #find length of triangle edges
    
    # a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    # b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    # c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    a = distance.euclidean(np.array(left_eye), np.array(point_3rd))
    b = distance.euclidean(np.array(right_eye), np.array(point_3rd))
    c = distance.euclidean(np.array(right_eye), np.array(left_eye))
    
    #-----------------------
    
    #apply cosine rule
    
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
    
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
    
        #-----------------------
        #rotate base image
    
        if direction == -1:
            angle = 90 - angle
    
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    
    #-----------------------
    
    return img #return img anyway



detector = MTCNN()
 
img = cv2.imread("/home/saad/saad/arcface/Face-Recognition_paul_arcface/data/processed/Sameed_elahi_93/Sameed_elahi_93_front_129_182_68_84_06182021_114147.png")

cv2.imshow("2",img)
cv2.waitKey(0)

detections = detector.detect_faces(img)
 
for detection in detections:
    score = detection["confidence"]
    if (score > 0.90):
       x, y, w, h = detection["box"]
       detected_face = img[int(y):int(y+h), int(x):int(x+w)]
    
    keypoints = detection["keypoints"]
    left_eye = keypoints["left_eye"]
    right_eye = keypoints["right_eye"]
    img = alignment_procedure(img, left_eye, right_eye)

cv2.imshow("1",img)
cv2.waitKey(0)

