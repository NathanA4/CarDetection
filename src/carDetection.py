import os
import cv2
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from track import Tracker

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    database=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)

cur = conn.cursor()

model = YOLO('yolov8s.pt')

stream = CamGear(source='https://www.youtube.com/watch?v=DnUFAShZKus', stream_mode=True, logging=True).start()

def insert_car_id(car_id):
    cur.execute("INSERT INTO car_ids (car_id) VALUES (%s)", (car_id,))
    conn.commit()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

my_file = open("src/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0
track = Tracker()

while True:
    frame = stream.read()
    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
            insert_car_id(d)

    bbox_idx = track.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
        cvzone.putTextRect(frame, f'{id1}', (x3, y3), 1, 1)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cur.close()
conn.close()
cv2.destroyAllWindows()
