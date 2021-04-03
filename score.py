import json
import numpy as np
import os
import pickle
import joblib
import pandas as pd

# called when the service is loaded
def init():
    global model
    model_path = os.path.join(os.environ['AZUREML_MODEL_DIR'],'model.joblib')
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    data = json.load(raw_data)['data']
    data = pd.DataFrame.from_dict(data)
    # get the prediction from the mode
    pred = model.predict(data)
    # return the predications as any JSON format
    return json.dumps({"result": pred.tolist()})

