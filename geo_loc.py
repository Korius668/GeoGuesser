import os
import cv2 as cv
from ultralytics import YOLO

def cut_signs(model, cap, output_dir):

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * 2
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * 2
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter("output_detected.mp4", fourcc, fps, (width, height))

    frame_count = 0
    object_id = 0

    classes = set(['different-traffic-sign', 'prohibition-sign', 'speed-limit-sign', 'warning-sign'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Folder '{output_dir}' został utworzony.")
    else:
        print(f"✅ Folder '{output_dir}' już istnieje.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        cloned_frame = frame.copy()

        if results.boxes is not None:
            boxes = results.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = box.conf.item()
                if class_name in classes and confidence > 0.5:
                    # Koordynaty bboxa
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw bounding box and label on the cloned frame
                    cv.rectangle(cloned_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv.putText(cloned_frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Wytnij obiekt
                    cropped = frame[y1:y2, x1:x2]

                    # Zapisz obrazek
                    filename = os.path.join(output_dir,
                                            f"frame{frame_count:04d}_{class_name}_{object_id}_{confidence:.2f}.jpg")
                    cv.imwrite(filename, cropped)
                    object_id += 1

        frame_count += 1
        out.write(cloned_frame)
    out.release()

if __name__ == "__main__":
    model = YOLO("models/fine_tuned_yolov8s.pt")
    cap = cv.VideoCapture("vid1.mov")
    output_dir = "detected_objects"
    cut_signs(model, cap, output_dir)
    cap.release()