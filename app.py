import numpy as np
import logging
from flask import Flask, request, jsonify, render_template
import pickle
import model
app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.ET_Model.predict(final_features)

    output = round(prediction[0], 1)

    return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))


if __name__ == "__main__":
    app.run(port=5000, debug=True)
    #app.run(debug=True)  # running the app