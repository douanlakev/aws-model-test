import os
import pickle
import json
import numpy as np
import logging
from flask import Flask, request, jsonify

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

app = Flask(__name__)

def model_fn(model_dir):
    """
    Load the model from the specified directory.
    """
    model_path = os.path.join(model_dir, 'xgb_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file does not exist: {model_path}')
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    logger.info("Model loaded successfully")
    return model

def input_fn(request_body, request_content_type='application/json'):
    """
    Process the input data from the request body.
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        features = np.array([input_data.get('floors', 0), input_data.get('lat', 0), input_data.get('long', 0)]).reshape(1, -1)

        return features
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make a prediction using the provided model and input data.
    """
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept='application/json'):
    """
    Format the prediction output as specified.
    """
    response = {'prediction': int(prediction[0])}
    if accept == 'application/json':
        return json.dumps(response), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

# Load the model
model_dir = '/opt/ml/model'
model = model_fn(model_dir)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Health check endpoint to verify if the model is loaded.
    """
    health = model is not None
    status = 200 if health else 404
    return jsonify({'status': 'Healthy' if health else 'Unhealthy'}), status

@app.route('/invocations', methods=['POST'])
def invoke():
    """
    Endpoint to process incoming requests and return predictions.
    """
    data = request.data.decode('utf-8')
    content_type = request.content_type
    
    # Process input data
    input_data = input_fn(data, content_type)
    
    # Make a prediction
    prediction = predict_fn(input_data, model)
    
    # Format the output
    response, content_type = output_fn(prediction, content_type)
    
    return response, 200, {'Content-Type': content_type}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)