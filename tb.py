from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from flask_cors import CORS  # Import CORS from flask_cors
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Serve index4.html when the root URL ('/') is accessed
@app.route('/')
def index():
    return render_template('index.html')



# Serve index2.html
@app.route("/tb-detector")
def tb_detector():
    return render_template("tb_detector.html")

@app.route("/symptoms")
def symptom():
    return render_template("symptoms.html")

@app.route("/diagnosis")
def diagnosis():
    return render_template("diagnosis.html")

@app.route("/prevention")
def prevention():
    return render_template("prevention.html")

@app.route("/treatment")
def treatment():
    return render_template("treatment.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")



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


# Connect the TB Detector form in index4.html to the result page in index2.html
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = 'temp_image.png'
        file.save(file_path)

        processed_image = preprocess_image(file_path)
        prediction = loaded_model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        result_message = "Normal" if predicted_class == 0 else "Abnormal - Suggestive Tuberculosis"

        # Add headers to enable CORS
        response = jsonify({'result': result_message})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Replace '*' with specific origins if needed

        return response

# Define an additional route for the diagnosis section in index2.html

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)