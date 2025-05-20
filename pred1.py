import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

class MonkeypoxPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.labels = ['Monkeypox', 'Non-Monkeypox']
        self.img_size = (224, 224)
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def predict_image(self, img_path):
        output_path = 'static/output_prediction.png'

        try:
            if os.path.exists(img_path):
                # Load and preprocess image
                with Image.open(img_path).convert('RGB') as img:
                    img = img.resize(self.img_size)
                    img_array = img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = self.model.predict(img_array)[0][0]
                pred_label = self.labels[int(prediction > 0.5)]
                prob = prediction if prediction > 0.5 else 1 - prediction

                # Save the output image with the prediction
                with Image.open(img_path).convert('RGB') as img:
                    plt.figure()
                    plt.imshow(img)
                    plt.title(f"{pred_label} ({prob * 100:.2f}%)")
                    plt.axis('off')
                    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                    plt.close()

                print(f"Prediction saved as {output_path}")
                return output_path, prob * 100

            else:
                print(f"Image not found: {img_path}")
                return None, None

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None, None
