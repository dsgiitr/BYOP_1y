import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template

#Creating the Flask App
app = Flask(__name__)

#Loading the Pickle Model
text_model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    text_features = [x for x in request.form.values()]
    #features = [np.array(text_features)]
    prediction = text_model.predict(text_features)

    return render_template("index.html", prediction_text = "The rating of the review is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
