import os
from PIL import Image
import pytesseract
from modules.language_ocr import detect_languages, detect_languages_by_path
from collections import Counter
from  modules.recognize_signs import procces_sig_image
# Define the directory containing detected objects
detected_objects_dir = "detected_objects"

# Lists to store images with and without text
images_with_text = []
images_without_text = []

# Iterate through all files in the directory
for file_name in os.listdir(detected_objects_dir):
    # Construct the full file path
    file_path = os.path.join(detected_objects_dir, file_name)

    # Check if the file is an image (e.g., ends with .jpg or .png)
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing image: {file_name}")

        # Open the image
        image = Image.open(file_path)

        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image, config='--psm 6').strip()

        # Check if text is present and the filename does not contain 'speed-limit-sign'
        if extracted_text and 'speed-limit-sign' not in file_name:
            images_with_text.append(file_name)
        else:
            images_without_text.append(file_name)

# Print the results
print("Images with text:", images_with_text)
print("Images without text:", images_without_text)

import re

# Iterate through images_with_text to find and move STOP signs or signs with just numbers
for file_name in images_with_text[:]:  # Use a copy of the list to avoid modifying it while iterating
    file_path = os.path.join(detected_objects_dir, file_name)
    image = Image.open(file_path)
    extracted_text = pytesseract.image_to_string(image).strip()

    # Check if the text contains "STOP" or is just numbers
    if "STOP" in extracted_text.upper() or re.fullmatch(r'\d+', extracted_text):
        images_with_text.remove(file_name)
        images_without_text.append(file_name)

# Print the updated results
print("Updated images with text:", images_with_text)
print("Updated images without text:", images_without_text)

images_with_text = [f"detected_objects/{text}" for text in images_with_text]
images_without_text = [f"detected_objects/{text}" for text in images_without_text]

print("Updated images with text:", images_with_text)


cities, languages = detect_languages_by_path(images_with_text)

print("languages", languages)

# Flatten the list of languages
all_languages = [lang for _, langs in languages for lang in langs]

# Count occurrences of each language
language_counts = Counter(all_languages)

# Convert to a list of tuples (language, count)
language_count_list = list(language_counts.items())

print("Counter plus lang", language_count_list)

decoded_signs = procces_sig_image(images_without_text)

print("Decoded signs:", decoded_signs)
