# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:51:03 2020

@author: adewole opeyemi
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import Net

app = Flask(__name__)
model = Net()
net.load_state_dict(torch.load(cifar_net.pth))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    '''Let's Predict Images
    This is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
        
    responses:
        200:
            description: The output values
    '''
    img = image_loader(request.files.get('file'))
    out = net(img.float())
    _, predicted = torch.max(out, 1)
    resp = classes[predicted]
    '''
    response = app.response_class(
        response=json.dumps(resp),
        status=200,
        mimetype='application/json'
      )'''
    
    return resp


if __name__ == '__main__':
    app.run(port=5002)    
    