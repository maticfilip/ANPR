import cv2
import easyocr
import csv
import datetime
import pytesseract

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

model = "haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500
capture_interval = 2  # seconds
last_capture_time = datetime.datetime.now() - datetime.timedelta(seconds=capture_interval)

csv_file_path = 'recognized_plates.csv'
img_path = "plates/scaned_img.jpg"  # Image will be overwritten each time

while True:
    success, img = cap.read()
    if not success:
        break

    current_time = datetime.datetime.now()
    if (current_time - last_capture_time).total_seconds() >= capture_interval:
        plate_model = cv2.CascadeClassifier(model)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_model.detectMultiScale(img_gray, 1.1, 4)

        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                
                img_roi = img[y:y+h, x:x+w]
                cv2.imwrite(img_path, img_roi)  # Save the ROI image, overwriting the existing image
                
                # Use EasyOCR to read text from the saved image
                results = reader.readtext(img_path)
                
                # Process EasyOCR results
                text = ' '.join([result[1] for result in results])
                
                # Write recognized text to CSV file
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([current_time.strftime('%Y-%m-%d %H:%M:%S'), img_path, text])
                
                last_capture_time = current_time  # Update the last capture time
                
                break  # Break after processing the first detected plate

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
