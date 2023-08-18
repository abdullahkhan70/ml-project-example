from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomDataClass, PredictPipeline

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Route for our Home Page.

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-data", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        custom_data_class = CustomDataClass(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )

        features = custom_data_class.get_custom_data()
        predict_pipeline = PredictPipeline()
        predict = predict_pipeline.predict(features=features)
        return render_template("home.html", results = predict[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000 ,debug=True)