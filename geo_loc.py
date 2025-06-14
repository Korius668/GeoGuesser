import os
import cv2 as cv
from ultralytics import YOLO
from IPython.display import Video
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

model = YOLO("models/fine_tuned_yolov8s.pt")

cap = cv.VideoCapture("vid1.mov")

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*"mp4v")  # lub 'XVID' dla AVI
out = cv.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

frame_count = 0
object_id = 0

output_dir = "detected_objects"
classes = set(['different-traffic-sign', 'prohibition-sign', 'speed-limit-sign', 'warning-sign'])
# Sprawdź czy folder istnieje, jeśli nie — stwórz
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 Folder '{output_dir}' został utworzony.")
else:
    print(f"✅ Folder '{output_dir}' już istnieje.")


# 4. Pętla po klatkach
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 5. Detekcja
    results = model(frame)[0]  # YOLOv8 returns list; take the first

    # Create a copy of the frame to draw filtered detections
    filtered_frame = frame.copy()

    if results.boxes is not None:
        boxes = results.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name in classes:
                # Koordynaty bboxa
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw the bounding box and label on the filtered frame
                cv.rectangle(filtered_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(filtered_frame, class_name, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Wytnij obiekt
                cropped = frame[y1:y2, x1:x2]

                # Zapisz obrazek
                filename = os.path.join("detected_objects", f"frame{frame_count:04d}_{class_name}_{object_id}.jpg")
                cv.imwrite(filename, cropped)
                object_id += 1

    # Write the filtered frame to the output video
    out.write(filtered_frame)
    frame_count += 1

# 8. Zwolnienie zasobów
cap.release()
out.release()