
from flask import Flask, request, jsonify, render_template,session
import numpy as np
import cv2
import os
import tensorflow as tf
from flask import Flask, request, jsonify
import vonage
from geopy.geocoders import Nominatim
import pyshorteners

app = Flask(__name__)
model_path = 'C:\\Users\\Pradip Sarkar\\Desktop\\garbage_detection_model .h5'  # Corrected model path
model = tf.keras.models.load_model(model_path)

def preprocess_image(img):
    # Preprocess the image (resize, normalize, etc.)
    img_array = cv2.resize(img, (224, 224))
    img_array = np.reshape(img_array, [-1, 224, 224, 3])
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    
    # Save the uploaded file to the current directory
    upload_path = file.filename
    file.save(upload_path)
    
    # Read the uploaded file using OpenCV
    image = cv2.imread(upload_path)
    image = preprocess_image(image)
    
    # Make predictions using the model
    prediction = np.argmax(model.predict(image))
    
    # Assuming binary classification 
    result = 'Organic' if prediction > 0 else 'Recyclable'

    return jsonify({'result': result})

# Nexmo credentials
client = vonage.Client(key="e71f36d9", secret="EfaMLIhX6VRZTEF8")
sms = vonage.Sms(client)


@app.route('/upload', methods=['POST'])
def upload():
    # Get image file and location data
    latitude = request.form['latitude']
    longitude = request.form['longitude']
    
    # Reverse geocode location data (replace this with your preferred method)
    location = f'Latitude: {latitude}, Longitude: {longitude}' 

    # Generate map link
    map_url = f'https://www.google.com/maps?q={latitude},{longitude}'
    
    # Shorten URL
    shortener = pyshorteners.Shortener()
    shortened_url = shortener.tinyurl.short(map_url)
    
    # Send SMS with location data
    message = f'User location: {location}. In this location garbage is found please collect it out!! Click here to view on map: {shortened_url}.'
    responseData = sms.send_message(
    {
        "from": "Vonage APIs",
        "to": "919330338139",
        "text": message,
    }
    )
    if responseData["messages"][0]["status"] == "0":
        return jsonify({'message': 'Location sent successfully'})
    else:
        return jsonify({'error': 'Failed to send location'})


if __name__ == '__main__':
    app.run(debug=True)


