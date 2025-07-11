from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            # Always pass results=None to avoid undefined template errors
            return render_template('home.html', results=None)
        else:
            # Note: swapped back to correct reading_score and writing_score
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            print("✅ Data object created")
            pred_df = data.get_data_as_data_frame()
            print("📊 DataFrame:", pred_df)

            predict_pipeline = PredictPipeline()
            print("🚀 Predict pipeline created")
            results = predict_pipeline.predict(pred_df)
            predicted_value = float(results[0])
            print("🎯 Prediction results:", predicted_value)

            return render_template('home.html', results=predicted_value)
    except Exception as e:
        print("🔥 ERROR:", e)
        return f"Internal Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0")
