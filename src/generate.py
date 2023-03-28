import cv2,os,sys


face_classifier = cv2.CascadeClassifier('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml')


def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)

count = 0
roll = sys.argv[1]
# roll = 140005


while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame),(200,200))
        # cv2.putText(face,str(count),(0,199),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,127),2)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cv2.circle(face, (182,180), 18, (0,0,0), -1)
        cv2.putText(face,str(count),(170,185),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
 
        file_name_path = 'C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset\\'+str(roll)+"."+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        
        cv2.imshow('Dataset',face) 
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!') 

