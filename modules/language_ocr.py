import os
from collections import Counter
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


import pytesseract
from easyocr import Reader
import re

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

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric characters and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading/trailing whitespace
    return text.lower()  # Convert to lowercase

def detect_languages_by_path(frame_paths):
    """
    Detects all languages present in text within PNG image frames using multiple EasyOCR readers
    for text extraction and langdetect for language identification.

    Args:
        frame_paths (list): A list of paths to the image frames (PNG files).

    Returns:
        list: A list of unique languages detected across all frames.
              Language names are human-readable (e.g., "English", "Japanese").
    """
    # Ensure frame_paths is a list
    if not isinstance(frame_paths, list):
        raise ValueError("frame_paths must be a list of file paths.")

    DetectorFactory.seed = 0
    detected_languages_list = []
    readable_texts = []
    import spacy  # Import spaCy for named entity recognition
    for file_path in frame_paths:
        print(f"\nProcessing file: {file_path}")  # Print current file being processed

        image_text_results = []

        # Read the image
        image = cv.imread(file_path)

        # Perform OCR to extract text
        text = pytesseract.image_to_string(image, config='--psm 6')

        # Clean and normalize the text
        readable_text = clean_text(text)

        # Filter out empty or very short strings
        if len(readable_text) > 2:  # Adjust the threshold as needed
            readable_texts.append(readable_text)

    from langdetect import detect_langs  # Import detect_langs for multiple language probabilities
    from langcodes import Language  # Import Language to convert codes to full names

    city_names = []
    language_info = []
    nlp = spacy.load("en_core_web_sm")

    for text in readable_texts:
        # Detect the 5 most probable languages
        probable_languages = detect_langs(text)
        top_languages = [Language.get(lang.lang).display_name() for lang in
                         probable_languages[:5]]  # Convert to full names
        language_info.append((text, top_languages))

        # Use spaCy to extract city names
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":  # GPE (Geopolitical Entity) often includes city names
                city_names.append(ent.text)

    return city_names, language_info

