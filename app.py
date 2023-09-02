import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
rftmodel=pickle.load(open('rft_model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    X = np.array(list(data.values())).reshape(1,-1)
    output=rftmodel.predict(X)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    output=rftmodel.predict(final_input)[0]
    if output == 1:
        output_string = "The red wine is of high quality."
    else:
        output_string = "The red wine is of low quality."
    return render_template("home.html",prediction_text=output_string)
    

if __name__=="__main__":
    app.run(debug=True)
    

