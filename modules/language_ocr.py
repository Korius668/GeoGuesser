import os
from collections import Counter
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from PIL import Image
import matplotlib.pyplot as plt

def clean_unconfident(ocr_response):
    i=0
    while i < len(ocr_response):
        if ocr_response[i][2]<0.5:
            del ocr_response[i]
        else: 
            i+=1
    return ocr_response 

def detect_languages(frame_dir, all_readers):
    """
    Detects all languages present in text within PNG image frames in a given directory
    using multiple EasyOCR readers for text extraction and langdetect for language identification.
    Addresses EasyOCR's compatibility requirement for Chinese models.

    Args:
        frame_dir (str): The path to the directory containing the image frames (PNG files).

    Returns:
        list: A list of unique languages detected across all frames.
              Language names are human-readable (e.g., "English", "Japanese").
    """
    DetectorFactory.seed = 0

    detected_languages_list = []
    print(os.listdir(frame_dir))
    for file_name in sorted(os.listdir(frame_dir)):
       
        if  file_name.lower().endswith(".jpg") or file_name.lower().endswith(".png"):
            file_path = os.path.join(frame_dir, file_name)
            print(f"\nProcessing file: {file_path}") # Print current file being processed

            image_text_results = []

            for i, reader in enumerate(all_readers):
                try:
                    results = reader.readtext(file_path)
                    
                    if results:
                        results = clean_unconfident(results)
                        if len(results)>0:
                            image_text_results.extend(results)
                except Exception as e:
                    print(f"  Error with Reader {i+1} on file {file_name}: {e}")
                    # Continue to the next reader even if one fails

            if image_text_results:
                for (_, text, confidence) in image_text_results:
                    try:
                        lang_code = detect(text)
                        detected_languages_list.append(lang_code)
                        print(f"  Detected text: '{text}' (Confidence: {confidence:.2f}) -> Language: {lang_code}")
                    except Exception as e:
                        print(f"  Error with text: {text} - {e}")
           
    lang_counts = Counter(detected_languages_list)
    
    return lang_counts.most_common(1)