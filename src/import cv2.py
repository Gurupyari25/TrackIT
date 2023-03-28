import cv2
import os
import sys
import numpy as np
from PIL import Image;


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


font = cv2.FONT_HERSHEY_SIMPLEX
face_id = sys.argv[1]
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml')
count = 0

assure_path_exists("C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset")
while (True):
    cv2.imshow('frame', image_frame)
    _, image_frame = cap.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x-2, y-15), (x + w+22, y + h+15), (9, 174, 255), 3)
        count += 1
        cv2.putText(image_frame, str(count), (x+165, y -25), font, 1, (9, 174, 255), 2)
        # Save the captured image into the datasets folder
        cv2.imwrite("C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset\\" + str(face_id) + '.' + str(count) + ".jpg", gray[y-15:y + h+15, x-2:x + w+22])
        # cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count >= 50:
        print("Successfully Captured")
        break

cap.release()
cv2.destroyAllWindows()


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml");

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[0])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset')
s = recognizer.train(faces, np.array(Ids))
print("Successfully trained")
recognizer.write('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\trainer.yml')
