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
output_dir = "license_plate_images3"
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

# Funkcija za OCR i procesiranje
def perform_ocr(image, save_prefix):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{save_prefix}_gray.jpg", gray)

    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{save_prefix}_resized.jpg", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(f"{save_prefix}_gaussian_blur.jpg", blur)

    gray = cv2.medianBlur(gray, 3)
    cv2.imwrite(f"{save_prefix}_median_blur.jpg", gray)

    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"{save_prefix}_threshold.jpg", thresh)

    #Segmentacije tablice 

    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kern, iterations=1)
    cv2.imwrite(f"{save_prefix}_dilation.jpg", dilation)

    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    plate_num = ""
    char_index = 0
    for cnt in sorted_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        height, width = gray.shape
        if height / float(h) > 6: continue
        ratio = h / float(w)
        if ratio < 1.5: continue
        area = h * w
        if width / float(w) > 15: continue
        if area < 100: continue
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        roi = cv2.medianBlur(roi, 5)
        
        # Spremanje svakog znaka
        char_save_path = f"{save_prefix}_char_{char_index}.jpg"
        cv2.imwrite(char_save_path, roi)
        char_index += 1
        
        text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10 --oem 3')
        plate_num += text.strip()
    return plate_num

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
                    recognized_text = perform_ocr(license_plate_img, img_prefix)
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