import cv2
import easyocr
import csv
import datetime

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

model = "haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500
count = 0

csv_file_path = 'recognized_plates.csv'

while True:
    success, img = cap.read()
    plate_model = cv2.CascadeClassifier(model)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_model.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_path = "plates/scaned_img_" + str(count) + ".jpg"
        cv2.imwrite(img_path, img_roi)
        
        results = reader.readtext(img_path)
        
        text = ' '.join([result[1] for result in results])
        
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write as [timestamp, image path, recognized text]
            writer.writerow([datetime.datetime.now(), img_path, text])
        
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break