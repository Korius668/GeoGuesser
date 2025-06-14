import os
import cv2 as cv
import pytesseract
import re
import spacy
from langdetect import detect
from langcodes import Language
from collections import Counter

try:
    spacy.cli.download("en_core_web_sm")
    print("Model downloaded successfully!")
except Exception as e:
    print(f"An error occurred: {e}")

# Path to the folder containing detected objects
input_folder = "detected_objects"

# Lists to store classified image filenames
signs_with_text = []
signs_without_text = []

# Function to check if text contains characters other than numbers
def contains_non_numeric(text):
    return any(char.isalpha() for char in text)

# Function to clean and normalize text
def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Keep only alphanumeric characters and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = text.strip()  # Remove leading/trailing whitespace
    return text.lower()  # Convert to lowercase

# List to store cleaned and readable texts
readable_texts = []

# Iterate through all images in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(input_folder, filename)

        # Read the image
        image = cv.imread(filepath)

        # Perform OCR to extract text
        extracted_text = pytesseract.image_to_string(image, config='--psm 6')

        # Classify based on the content of the text
        if contains_non_numeric(extracted_text):
            signs_with_text.append(filename)
        else:
            signs_without_text.append(filename)

print("Signs with text:", signs_with_text)
print("Signs without text:", signs_without_text)

# List to store extracted texts
extracted_texts = []

# Iterate through the signs_with_text list
for filename in signs_with_text:
    filepath = os.path.join(input_folder, filename)

    # Read the image
    image = cv.imread(filepath)

    # Perform OCR to extract text
    text = pytesseract.image_to_string(image, config='--psm 6')

    # Clean and normalize the text
    readable_text = clean_text(text)

    # Filter out empty or very short strings
    if len(readable_text) > 2:  # Adjust the threshold as needed
        readable_texts.append(readable_text)

print("Extracted texts in readable format:", readable_texts)


# Load spaCy's pre-trained NER model
nlp = spacy.load("en_core_web_sm")

# Function to extract city names and detect language
def analyze_texts(texts):
    from langdetect import detect_langs  # Import detect_langs for multiple language probabilities
    from langcodes import Language  # Import Language to convert codes to full names

    city_names = []
    language_info = []

    for text in texts:
        # Detect the 5 most probable languages
        probable_languages = detect_langs(text)
        top_languages = [Language.get(lang.lang).display_name() for lang in probable_languages[:5]]  # Convert to full names
        language_info.append((text, top_languages))

        # Use spaCy to extract city names
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "GPE":  # GPE (Geopolitical Entity) often includes city names
                city_names.append(ent.text)

    return city_names, language_info

# Analyze the readable texts
cities, languages = analyze_texts(readable_texts)

print("Identified city names:", cities)
print("Detected languages:", languages)
