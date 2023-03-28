from cProfile import label
from datetime import datetime
from re import X
import cv2,os,sys
import numpy as np
import pymongo
from os import listdir
from os.path import isfile, join


data_path = 'C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset\\'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
date = datetime.now().strftime("%d/%m/%Y")
Training_Data, Labels = [], []
dict={
    "Date": date
}

Name ={

}
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(int(os.path.split(files)[1].split(".")[0]))

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["trackit"]
mycol  = mydb["students"]
mycol1 = mydb["mlstatuses"]
mycol2 = mydb["cnstatuses"]
mycol3 = mydb["iotstatuses"]

face_classifier = cv2.CascadeClassifier('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml')


def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.1,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

req = sys.argv[1]

ids = mycol.find({},{"id": 1, "name": 1, "_id": 0})
for i in ids:
     op={str(i["id"]):str(i["name"])}
     Name.update(op)



for i in Labels:
     z={str(i):"Absent"}
     dict.update(z)







cap = cv2.VideoCapture(0)




while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)
        

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))
            
        if confidence > 78:
            # for t in Name:
            #     name = Name[str(result[0])]
            cv2.rectangle(image,(0,420),(150,478),(0,255,255),-1)
            cv2.putText(image,str(result[0]),(5,445), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            cv2.putText(image,str(result[0]),(5,474), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            cv2.imshow('Face Cropper', image)
            if(str(result[0] not in dict)):
                dict[str(result[0])] = "Present"

            


        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()
if(req == "ML"):
    mycol1.insert_one(dict)
elif(req == "CN"):
    mycol2.insert_one(dict)
else:
    mycol3.insert_one(dict)