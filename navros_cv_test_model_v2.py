from ultralytics import YOLO
import cv2
import math 

width = 640 
height = 480

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# model
model = YOLO("weights/best.pt")

# object classes
classNames = ["cone"]


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # // for integer division
    # 1st Quadrant
    quadrants = {
    "q1": [(0, 0), (width//2, height//2)],
    "q2": [(width//2, 0), (width, height//2)],
    "q3": [(0, height//2), (width//2, height)],
    "q4": [(width//2, height//2), (width, height)]
    }

    cv2.rectangle(img, quadrants[key][0], quadrants[key][1], (0, 255, 0), 2) 
    # 2nd Quadrant
    cv2.rectangle(img, (width//2, 0), (width , height//2), (255, 255, 0), 2) 
    # 3rd Quadrant
    cv2.rectangle(img, (0, height//2), (width//2, height), (0, 0, 255), 2)
    # 4th Quadrant 
    cv2.rectangle(img, (width//2, height//2), (width, height), (255, 0, 0), 2) 
    quadrants = {q1: (0,0)}

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            bb_center = [(x1+x2)/2, (y1+y2)/2]

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()