from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomPredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            # Web form data
            data = CustomPredictPipeline(
                gender=request.form['gender'],
                race_ethnicity=request.form['race_ethnicity'],
                parental_level_of_education=request.form['parental_level_of_education'],
                lunch=request.form['lunch'],
                test_preparation_course=request.form['test_preparation_course'],
                reading_score=int(request.form['reading_score']),
                writing_score=int(request.form['writing_score'])
            )

            pred_df = data.get_data_as_dataframe()
            prediction_pipeline = PredictPipeline()
            prediction, model_name = prediction_pipeline.predict(pred_df)

            return render_template('home.html', prediction=prediction, model_name=model_name)

        except Exception as e:
            return jsonify({'error': str(e)}), 500


# ✅ API for Postman
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        json_data = request.get_json()

        data = CustomPredictPipeline(
            gender=json_data['gender'],
            race_ethnicity=json_data['race_ethnicity'],
            parental_level_of_education=json_data['parental_level_of_education'],
            lunch=json_data['lunch'],
            test_preparation_course=json_data['test_preparation_course'],
            reading_score=int(json_data['reading_score']),
            writing_score=int(json_data['writing_score'])
        )

        pred_df = data.get_data_as_dataframe()
        prediction_pipeline = PredictPipeline()
        prediction, model_name = prediction_pipeline.predict(pred_df)

        return jsonify({
            'prediction': prediction,
            'model_used': model_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
