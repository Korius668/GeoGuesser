import os
import sys
import cv2 as cv
from ultralytics import YOLO
from modules.language_ocr import detect_languages
from modules.driving_side import analyze_driving_side
import pandas as pd
from modules.mark_on_map import mark_multiple_countries_with_distances


def cut_signs(model, cap, output_dir):

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
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
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path) # Remove file or symbolic link
                    print(f"  Removed file: {item}")
                
            except OSError as e:
                print(f"  Error removing {item}: {e}")


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
                
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv.rectangle(cloned_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv.putText(cloned_frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
                    cropped = frame[y1:y2, x1:x2]
                    filename = os.path.join(output_dir,
                                            f"frame{frame_count:04d}_{class_name}_{object_id}_{confidence:.2f}.jpg")
                    cv.imwrite(filename, cropped)
                    object_id += 1
        frame_count += 1
        out.write(cloned_frame)
    out.release()

if __name__ == "__main__":
    output_dir = ""
    try:
        if len(sys.argv) < 3:
            print("Użycie: python geo_loc.py <lokalizacja_video> <aktualne_współrzędne>")
            sys.exit(1)

        video_path = sys.argv[1]
        actual_coords = tuple(map(float, sys.argv[2].strip("()").split(",")))

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Plik wideo nie istnieje: {video_path}")

        try:
            model = YOLO("models/fine_tuned_yolov8s.pt")
        except Exception as e:
            raise RuntimeError(f"Nie udało się załadować modelu YOLO: {e}")

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Nie można otworzyć pliku wideo: {video_path}")

        output_dir = "detected_objects"

        try:
            cut_signs(model, cap, output_dir)
        except Exception as e:
            raise RuntimeError(f"Błąd podczas wycinania znaków: {e}")

    except Exception as err:
        print(f"[BŁĄD] {type(err).__name__}: {err}")
    finally:
        try:
            cap.release()
        except NameError:
            pass

    if output_dir and video_path:
        languages = detect_languages(output_dir)
        driving_site = analyze_driving_side(video_path)
        print(f"Detected languages: {languages}")
        print(f"Driving side: {driving_site}")
        countries_df = pd.read_csv('countries.csv')

        #filter countries based on detected languages and driving side
        filtered_countries = countries_df[
            (countries_df['language-short'].isin(languages)) &
            (countries_df['driving-direction'] == driving_site)
        ]

        print(f"Filtered countries based on languages and driving side: {filtered_countries}")

        if not filtered_countries.empty:
            filtered_countries_codes = filtered_countries['country-code'].tolist()
            mark_multiple_countries_with_distances(actual_coords, filtered_countries_codes)
        else:
            print("No countries match the detected languages and driving side.")