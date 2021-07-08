from retinaface import RetinaFace
import glob
import cv2

# /home/saad/saad/arcface/Face-Recognition_paul_arcface/Not_same_images/retina_face_plotted
#'/home/saad/saad/arcface/Face-Recognition_paul_arcface/Not_same_images/retina_face/11_orignal_name_Khalid_Babar_118_Predicted_name:Khalid_Babar_123999.png'


def convertTuple(tup):
    str =  '_'.join(tup)
    return str

    
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



name_list1 = []
count = 0
# photos = glob.glob('/home/saad/saad/arcface/insightface/recognition/arcface_torch/data/processed/*')
photos = glob.glob('/home/saad/saad/arcface/insightface/recognition/arcface_torch/data/test_images/Abdul_basit_17/*')
saving_path = "/home/saad/saad/arcface/insightface/recognition/arcface_torch/data/aligned_faces/"

for p in photos:
    full = p
    str1 = p.split('/')[-1].replace(' ', '').replace('right', '').replace('random', '').replace('left', '').replace('front', '').replace('.jpg', '')
    str2 = str1.split('_')[:3]
    str3 = convertTuple(str2)
    name_list1.append(str3)

   
for elem in enumerate(photos):
    
    print(elem[0])
    #######################################################################

    photo = cv2.imread(elem[1])
    obj = RetinaFace.detect_faces(photo)
    # print(obj.keys())
    
    try:
        for key in obj.keys():   

                identity = obj[key]
                # print(identity)

            ################## MAPPING LANDMARKS ##################

                landmarks = identity['landmarks']
                re = landmarks['right_eye']
                le = landmarks['left_eye']
                n = landmarks['nose']
                rm = landmarks['mouth_right']
                lm = landmarks['mouth_left']
                
                photo = alignment_procedure(photo, le, re)
                
                # cv2.circle(photo, (re[0], re[1]), 3, (168, 0, 20), -1)
                # cv2.circle(photo, (le[0], le[1]), 3, (168, 0, 20), -1)
                # cv2.circle(photo, (n[0], n[1]), 3, (168, 0, 20), -1)
                # cv2.circle(photo, (rm[0], rm[1]), 3, (168, 0, 20), -1)
                # cv2.circle(photo, (lm[0], lm[1]), 3, (168, 0, 20), -1)

            ################## MAPPING BOUNDING BOXES ##################
                # bb = identity['facial_area']
                # cv2.rectangle(photo,(bb[2],bb[3]),(bb[0],bb[1]),(255,255,255),1)
                
            ################## SAVING IMAGES ##################
                # cv2.imwrite("/home/saad/saad/arcface/Face-Recognition_paul_arcface/Not_same_images_retina_plot/retina_face_plotted/"+str(elem[0])+".png",photo)
                cv2.imwrite(saving_path+""+name_list1[elem[0]]+"_"+str(elem[0])+".png",photo)
                break

            #######################################################################
        
    except:
        # print(elem[1])
        count = count + 1
        pass 

print("total missed: ",count)
