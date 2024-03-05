from flask import Flask, render_template, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model_path = 'D:\\Final Year project\\metadata\\models\\model.h5'
loaded_model = load_model(model_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    
    return img.reshape(1, 28, 28, 3)

@app.route('/')
def home():
    return render_template('index3.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = 'temp_image.png'
        file.save(file_path)

        processed_image = preprocess_image(file_path)
        prediction = loaded_model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        result_message = "Not Tuberculosis" if predicted_class == 0 else "Tuberculosis"

        # Add headers to enable CORS
        response = jsonify({'result': result_message})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Replace '*' with specific origins if needed

        return response

if __name__ == '__main__':
    app.run(debug=True)



