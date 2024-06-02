from ultralytics import YOLO
import cv2
import math 
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd="C:/Users/Perry Hotter/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
output_dir = "license_plate_images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("best.pt")

classNames = ["license-plate"]

min_confidence_threshold = 0.5  
save_image = False
saved_count = 0

def show_notification(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Image Saved", message)
    root.destroy()

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(enlarged)
    
    denoised = cv2.fastNlMeansDenoising(contrast_enhanced, None, h=5)
    
    _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    return threshold
    

def perform_ocr(image):
    text=pytesseract.image_to_string(image, lang ='eng', 
    config ='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return text

def save_text_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

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

                if save_image:
                    license_plate_img = img[y1:y2, x1:x2]
                    preprocessed_img = preprocess_image(license_plate_img)
                    cv2.imwrite(os.path.join(output_dir, f"license_plate_{saved_count + 1}.jpg"), license_plate_img)

                    recognized_text = perform_ocr(license_plate_img)

                    save_text_to_file(recognized_text, os.path.join(output_dir, f"license_plate_{saved_count + 1}.txt"))

                    saved_count += 1
                    show_notification("License plate image saved successfully.")
                    save_image = False

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_image = True

cap.release()
cv2.destroyAllWindows()
