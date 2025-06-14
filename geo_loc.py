import cv2 as cv
from ultralytics import YOLO
from IPython.display import Video
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns

model = YOLO("models/fine_tuned_yolov8s.pt")

cap = cv.VideoCapture("film.mp4")

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*"mp4v")  # lub 'XVID' dla AVI
out = cv.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

# 4. Pętla po klatkach
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 5. Detekcja
    results = model(frame)[0]  # YOLOv8 returns list; take the first

    # 6. Rysowanie wykryć
    annotated_frame = results.plot()  # rysuje boxy, etykiety itd.

    # 7. Zapis klatki
    out.write(annotated_frame)

# 8. Zwolnienie zasobów
cap.release()
out.release()