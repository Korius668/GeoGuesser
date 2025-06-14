from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions

def procces_sig_image(img_paths):
    decode_predictions_text = []
    for img_path in img_paths:
        model = MobileNetV3Small(weights='imagenet')

        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)

        # Decode predictions
        print(decode_predictions(preds, top=3)[0])
        decode_predictions_text.append(decode_predictions(preds, top=3)[0])
    return decode_predictions_text

