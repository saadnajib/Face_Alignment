from mtcnn import MTCNN
import cv2
from scipy.spatial import distance
import numpy as np
import math 
from PIL import Image
import glob
 


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



saving_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/data/aligned_faces/"
image_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/data/test_images/Arshad_Hasim_110/"

def convertTuple(tup):
    str =  '_'.join(tup)
    return str



images = glob.glob(image_path+'*')
full_path1 = []
name_list1 = []



for p in images:
    full = p
    str1 = p.split('/')[-1].replace(' ', '').replace('right', '').replace('random', '').replace('left', '').replace('front', '').replace('.jpg', '')
    str2 = str1.split('_')[:3]
    str3 = convertTuple(str2)
    full_path1.append(full)
    name_list1.append(str3)



for elem in enumerate(full_path1):
    
    print(elem[0])
    detector = MTCNN()
    img = cv2.imread(elem[1])

    # cv2.imshow("2",img)
    # cv2.waitKey(0)

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

    cv2.imwrite(saving_path+""+name_list1[elem[0]]+"_"+str(elem[0])+".png",img)
    # cv2.imshow("1",img)
    # cv2.waitKey(0)

