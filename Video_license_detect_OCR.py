import cv2
import easyocr
from yolov8 import YOLOv8
import os
import numpy as np
import csv
from cap_from_youtube import cap_from_youtube

# Initialize video
# videoUrl = "https://youtu.be/iUF7WJXaapM?feature=shared"
# cap = cap_from_youtube(videoUrl, resolution="720p")
video_capture = cv2.VideoCapture('test_images/video_1.mp4')

 # skip first {start_time} seconds
# video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_capture.get(cv2.CAP_PROP_FPS))

# Initialize YOLOv7 model
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Create a directory to save binary images
output_dir = "binary_images"
os.makedirs(output_dir, exist_ok=True)

# Create a list to store OCR results
ocr_results = []

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while video_capture.isOpened():
    # Press key q to stop
    if cv2.waitKey(1) == ord("q"):
        break

    try:
        # Read frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)

    # Extract text using EasyOCR for each bounding box
    for i, box in enumerate(boxes):
        x, y, w, h = map(int, box)  # Convert coordinates to integers
        roi = frame[y:h, x:w]

        # Use EasyOCR to do OCR on the region
        results = reader.readtext(roi)

        # Extract text from EasyOCR results
        text = ' '.join([result[1] for result in results])

        # Save the binary image
        binary_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)

        binary_filename = os.path.join(output_dir, f"binary_image_{i + 1}.jpg")
        cv2.imwrite(binary_filename, binary_image)
        # print(f"Binary image saved: {binary_filename}")

        # Store OCR results in a list
        ocr_results.append({'Box': i + 1, 'Text': text})

        # Draw the bounding box and text on the frame
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with bounding boxes and OCR text
        cv2.imshow("Detected Objects", frame)

# Store OCR results in a CSV file
csv_filename = "ocr_results.csv"
fields = ['Box', 'Text']
import pandas as pd
# ocr_results=pd.Dataframe()
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=fields)
    csv_writer.writeheader()
    csv_writer.writerows(ocr_results)

print(f"OCR results and binary images are stored in {csv_filename} and {output_dir}")
cv2.destroyAllWindows()
