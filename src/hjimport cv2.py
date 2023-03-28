hjimport cv2
import numpy as np
import xlwrite
import time
import sys
from datetime import datetime
import pymongo
 
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myclient["trackit"]
stud = db["students"]
ml = db["mlstatuses"]
cn = db["cnstatuses"]
iot = db["iotstatuses"]

start = time.time()
period = 8

casclf = cv2.CascadeClassifier('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\haarcascade.xml')

cap = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\gurup\\OneDrive\\Desktop\\trackIT\\src\\trainer.yml')

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

font = cv2.FONT_HERSHEY_SIMPLEX 

while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = casclf.detectMultiScale(gray, 1.1, 5);
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        
        roll, conf = recognizer.predict(roi_gray)
        
        if (conf < 70):

             if(roll):
                if(dict[str(roll)] == "Absent"):
                    dict[str(roll)] = "Present"
                
                studData = stud.find({"id": roll},{"_id": 0})
                for item in studData:
                    sid = item['id']
                    name = item['name'] 
                    course = item['course']
                    sem = item['semester']

                cv2.rectangle(img, (x, y-20), (x+w, y+h+20), (9, 174, 255), 2);
                cv2.rectangle(img, (x, y+h+20),(x+w, y+h+80), (9, 174, 255), cv2.FILLED)
                cv2.putText(img, str(sid) , (x+2, y+h+40), font, 0.5, (0,0,0), 2)
                cv2.putText(img,  name , (x+2, y+h+55), font, 0.5, (0,0,0), 2)
                cv2.putText(img,  course + " - " +sem , (x+2, y+h+70), font, 0.5, (0,0,0), 2)

        
    
    cv2.namedWindow("Reading faces", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Reading faces',1350, 710)
    cv2.imshow('Reading faces', img)

    if flag == 10:
        print("Transaction Blocked")
        break

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(req)
if(req == "ml"):
    ml.insert_one(dict)
elif(req == "cn"):
    cn.insert_one(dict)
else:
    iot.insert_one(dict)

print(type(roll))

print(dict)

