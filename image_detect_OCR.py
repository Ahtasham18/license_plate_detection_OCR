import cv2
import easyocr
from yolov8 import YOLOv8
import csv

# Initialize yolov8 object detector
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.3)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Read image
img = cv2.imread('./test_images/h1.png')

# Detect Objects
boxes, scores, class_ids = yolov8_detector(img)

# Create a copy of the image to draw on
print(boxes)
img_with_boxes = img.copy()
ocr_results = []
# Extract text using EasyOCR for each bounding box
for i, box in enumerate(boxes):
    x, y, w, h = map(int, box)  # Convert coordinates to integersq
    roi = img[y:h, x:w]

    
    # Use EasyOCR to do OCR on the region
    results = reader.readtext(roi)

    # Extract text from EasyOCR results
    text = ' '.join([result[1] for result in results])
    
    ocr_results.append({'Box': i + 1, 'Text': text})
    # Print the detected text
    print(f"Text in bounding box {i + 1}: {text}")

    # Draw the bounding box and text on the image
    cv2.rectangle(img_with_boxes, (x, y), (w,h), (0, 255, 0), 1)
    cv2.putText(img_with_boxes, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display image with detection
cv2.namedWindow("Detected Objects with OCR", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects with OCR", img_with_boxes)
cv2.imwrite("doc/img/detected_objects_with_ocr.jpg", img_with_boxes)
cv2.waitKey(0)

for i, box in enumerate(boxes):
    x, y, w, h = map(int, box)  # Convert coordinates to integersq
    roi = img[y:h, x:w]
    gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Set a threshold value 
    threshold_value = 100

    # Apply binary thresholding
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("OCR Text", cv2.WINDOW_NORMAL)
    cv2.imshow("OCR Text", gray_image)
    cv2.imshow("OCR Text", binary_image)
    cv2.waitKey(0)


# Store OCR results in a CSV file
csv_filename = "license.csv"
fields = ['Box', 'Text']

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=fields)
    csv_writer.writeheader()
    csv_writer.writerows(ocr_results)

print(f"OCR results are stored in {csv_filename}")
# [[628.20715 603.0273  813.30347 691.3349 ]]
# 633   604
# 812   601
# 634   690
# 814   689q