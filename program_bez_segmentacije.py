from ultralytics import YOLO
import cv2
import math 
import os
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pytesseract

# Tesseract putanja
pytesseract.pytesseract.tesseract_cmd = "C:/Users/Perry Hotter/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

# Izlazni folder
output_dir = "license_plate_images2"
os.makedirs(output_dir, exist_ok=True)

# Pokretanje web kamere
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Učitavanje YOLO modela
model = YOLO("best.pt")
classNames = ["license-plate"]
min_confidence_threshold = 0.5  
save_image = False
saved_count = 0

# Funkcija za prikaz obavijesti
def show_notification(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Image Saved", message)
    root.destroy()

# Funkcija za spremanje slike sa korakom obrade
def save_processing_step(img, step_name):
    step_dir = os.path.join(output_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)
    cv2.imwrite(os.path.join(step_dir, f"{step_name}_{saved_count + 1}.jpg"), img)

# Funkcija za OCR i procesiranje
def perform_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_processing_step(gray, "1_gray")

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    save_processing_step(gray, "2_resized")

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    save_processing_step(blur, "3_gaussian_blur")

    gray = cv2.medianBlur(gray, 3)
    save_processing_step(gray, "4_median_blur")

    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    save_processing_step(thresh, "5_threshold")

    text = pytesseract.image_to_string(thresh, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
    return text.strip()

# Spremanje pročitanog teksta
def save_text_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

# Glavna petlja
while True:
    success, img = cap.read()
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

                if save_image:
                    license_plate_img = img[y1:y2, x1:x2]
                    img_prefix = os.path.join(output_dir, f"license_plate_{saved_count + 1}")
                    cv2.imwrite(f"{img_prefix}.jpg", license_plate_img)
                    recognized_text = perform_ocr(license_plate_img)
                    save_text_to_file(recognized_text, f"{img_prefix}.txt")
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
