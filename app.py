from flask import Flask, render_template, request
#import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=['POST'])
def result():
    hypertension = float(request.form['hypertension'])
    heart_disease = float(request.form['heart_disease'])
    age = float(request.form['age'])
    avg_glucose_level=float(request.form['avg_glucose_level'])
    bmi=float(request.form['bmi'])
    x = np.array([hypertension, heart_disease, age,avg_glucose_level,bmi]).reshape(1, -1)
    
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    Y_pred = pickled_model.predict(x)
    res=Y_pred[0]
    #return render_template('index.html', prediction_text='Stroke Predection (normalized value) : {}'.format(res))
    if res==0:
      return render_template('index.html', prediction_text='Non-Stroke')
    else: 
      return render_template('index.html', prediction_text='"Stroke --> at Very Risk!!! , Immedietly consult the doctor')


if __name__ == "__main__":
    app.run(debug=True, port=7385)
