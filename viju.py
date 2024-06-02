from ultralytics import YOLO
import cv2
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("best-small.pt")
classNames = ["license-plate"]

min_confidence_threshold = 0.5

prev_frame_time = 0
new_frame_time = 0

while True:
    success, img = cap.read()
    
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = math.ceil((box.conf[0] * 100)) / 100

            if confidence >= min_confidence_threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cls = int(box.cls[0])
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.putText(img, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
