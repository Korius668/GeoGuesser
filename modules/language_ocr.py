import os
from collections import Counter
from langdetect import detect, DetectorFactory
from PIL import Image
import easyocr

other_langs = ['fr', 'de', 'en']

def clean_string(text):
    cleaned_chars = []
    for char in text:
        if char.isalpha() or char.isspace():
            cleaned_chars.append(char)
    return "".join(cleaned_chars)

def clean_unconfident(ocr_response, lang):
    response = []
    for word in ocr_response:
        if word[2]<0.5 or word[1]=="STOP":
            continue
        
        w = clean_string(word[1])
        if lang in ['chinese', 'japanese', 'korean'] or len(w)>3:
            response.append(w)
        
    return " ".join(response)

def detect_languages(frame_dir):
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
    try:
        reader_ch_tra = easyocr.Reader(['ch_tra'], gpu=True)
        reader_ch_sim = easyocr.Reader(['ch_sim'], gpu=True)
        reader_ja = easyocr.Reader(["ja"], gpu=True)
        reader_ko = easyocr.Reader(['ko'], gpu=True)
        reader_ru = easyocr.Reader(['ru'], gpu=True)
        reader_ar = easyocr.Reader(['ar'], gpu=True)
        reader_others = easyocr.Reader(other_langs, gpu=True)
        print("EasyOCR reader with GPU.")
    except Exception as e:
        print(f"Warning: GPU error for ch_tra_en reader. Falling back to CPU. Error: {e}")
        reader_ch_tra = easyocr.Reader(['ch_tra'], gpu=False)
        reader_ch_sim = easyocr.Reader(['ch_sim'], gpu=False)
        reader_ja = easyocr.Reader(["ja"], gpu=False)
        reader_ko = easyocr.Reader(['ko'], gpu=False)
        reader_ru = easyocr.Reader(['ru'], gpu=False)
        reader_ar = easyocr.Reader(['ar'], gpu=False)
        reader_others = easyocr.Reader(other_langs, gpu=False)

    all_readers = [reader_ch_tra, reader_ch_sim, reader_ja, reader_ko, reader_ru, reader_ar, reader_others]
    
    detected_languages_list = []

    for file_name in sorted(os.listdir(frame_dir)):
       
        if  file_name.lower().endswith((".jpg", ".png", '.jpeg')) and file_name.lower().count("different-traffic-sign")==1:
            file_path = os.path.join(frame_dir, file_name)

            image_text_results = set()
            image = Image.open(file_path)
            for i, reader in enumerate(all_readers):
                try:
                    
                    results = reader.readtext(image)
                    
                    if results:
                        resstr = clean_unconfident(results, reader.model_lang)
                        if resstr!=" ":
                            image_text_results.add(resstr)                    
                except Exception as e:
                    print(f"  Error with Reader {i+1} on file {file_name}: {e}")
                
            if image_text_results:
                for text in image_text_results:
                    try:                        
                        lang_code = detect(text)
                        detected_languages_list.append(lang_code)
                    except Exception as e:
                        print(f"  Error with text: {text} - {e}")
           
    lang_counts = Counter(detected_languages_list)
    
    
    return lang_counts.most_common(5) 