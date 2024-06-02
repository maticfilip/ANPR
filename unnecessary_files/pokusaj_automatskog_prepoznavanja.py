from ultralytics import YOLO
import cv2
import math 
import os
import time 
import tkinter as tk
from tkinter import messagebox

output_dir = "license_plate_images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model=YOLO("best.pt")

classNames=["license-plate"]

min_confidence_threshold = 0.5  
last_save_time = 0

def show_notification(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Image Saved", message)
    root.destroy()

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            confidence = math.ceil((box.conf[0]*100))/100

            if confidence >= min_confidence_threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                cls = int(box.cls[0])

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                current_time = time.time()
                if current_time - last_save_time >= 5:
                    license_plate_img = img[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(output_dir, f"license_plate_{len(os.listdir(output_dir)) + 1}.jpg"), license_plate_img)
                    last_save_time = current_time  
                    show_notification("License plate image saved successfully.")

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
