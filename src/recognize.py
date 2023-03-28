import cv2
import numpy as np
import os
import time
import sys
from datetime import datetime
import pymongo
from os import listdir
from os.path import isfile, join

data_path = 'C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\public\\dataset\\'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(int(os.path.split(files)[1].split(".")[0]))

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model training is successfully completed!")
 
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["trackit"]
stud = db["students"]
ml = db["mlstatuses"]
cn = db["cnstatuses"]
iot = db["iotstatuses"]

casclf = cv2.CascadeClassifier('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml')

cap = cv2.VideoCapture(0)

name = ""
course = ""
sem = ""
sid=0
flag = 0
roll = 0
date = datetime.now().strftime("%d/%m/%Y")
dict = {
    'Date': date,
    "140001": "Absent",
    "140002": "Absent",
    "140003": "Absent",
    "140004": "Absent",
    "140005": "Absent"
}
studData = []
req= sys.argv[1]

font = cv2.FONT_HERSHEY_DUPLEX

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = casclf.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (200,200))
        
        result = model.predict(roi_gray)
        roll = result[0]

        if result[1] < 500:
            conf = int(100*(1-(result[1])/300))
        
        if (conf > 80):
            if(dict[str(roll)] == "Absent"):
                dict[str(roll)] = "Present"
       
            studData = stud.find({"id": roll},{"_id": 0})
            for item in studData: 
                sid = item['id']
                name = item['name'] 
                course = item['course']
                sem = item['semester']

            cv2.rectangle(img, (x+10, y-10), (x+w-10, y+h+10), (9, 174, 255), 2)
            cv2.rectangle(img, (x+10, y+h+10),(x+w-10, y+h+90), (9, 174, 255), cv2.FILLED)
            cv2.putText(img, str(sid) , (x+15, y+h+30), font, 0.6, (0,0,0), 2)
            cv2.putText(img,  name , (x+15, y+h+55), font, 0.6, (0,0,0), 2)
            cv2.putText(img,  course + "-" +sem , (x+15, y+h+80), font, 0.6, (0,0,0), 2)      
    
    cv2.imshow('Taking Attendance', img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

if(req == "ml"):
    ml.insert_one(dict)
elif(req == "cn"):
    cn.insert_one(dict)
else:
    iot.insert_one(dict)

