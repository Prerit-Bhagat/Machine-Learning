from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import os

app = Flask(__name__)

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to your model file
model_file = 'modelpredict.h5'
model_path = os.path.join(current_dir, model_file)

# Check if the file exists and load the model
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")

    @app.route('/')
    def hello():
        return "Hello, welcome to the image prediction API!"

    @app.route('/image', methods=['POST'])
    def image():
        try:
            # Get pixels from the POST request
            pixels = request.form.get('pixels')
            
            # Convert the input to a NumPy array and reshape it
            # pixels = np.fromstring(pixels, sep=',')
            pixels=np.array([pixels])
            # Make prediction
            prediction = model.predict(pixels)
            
            # Apply softmax to get probabilities
            prediction = tf.nn.softmax(prediction)
            
            # Determine the predicted class
            val = np.argmax(prediction, axis=1)[0]
            
            # Prepare the response
            result = {'Image': int(val)}
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

    if __name__ == '__main__':
        app.run(debug=True, port=8000)
else:
    print(f"Error: The file '{model_file}' does not exist in the current directory.")
