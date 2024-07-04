from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import os

app = Flask(__name__)


# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to your model file
model_file = 'modelpredict.h5'
model_path = os.path.join(current_dir, model_file)

# Check if the file exists
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
    @app.route('/')
    def Hello():
        return "render_template"
    
    @app.route('/predict', methods=['POST','GET'])
    def predict():
        # Get data from POST request
        # data = request.get_json()
        
        # Preprocess the data (if needed)
        # Example: Convert JSON data to numpy array
        # x = np.array(data['input'])
        
        # Make prediction
        # predictions = model.predict(x)
        
        # Format prediction output
        # predicted_classes = np.argmax(predictions, axis=1)
        
        # Prepare response as JSON
        # response = {
            # 'predictions': predicted_classes.tolist()
        # }
        
        # return jsonify(response)
        return "Helo"
    if __name__=='__main__':
        app.run(debug=True,port=8000)
else:
    print(f"Error: The file '{model_file}' does not exist in the current directory.")






