import json
import numpy as np
import os
import pickle
import joblib

# called when the service is loaded
def init():
    global model
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'],'model.pkl')
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    data = np.array(json.load(raw_data)['data'])
    # get the prediction from the mode
    pred = model.predict(data)
    # return the predications as any JSON format
    return pred.to_list()

