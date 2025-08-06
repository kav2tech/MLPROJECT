from flask import Flask, request, render_template
import logging
import sys
import traceback

from src.pipeline.predict_pipeline import PredictPipeline, CustomPredictPipeline

# ----------------------------
# Flask App Setup
# ----------------------------
application = Flask(__name__)
app = application

# Logging Setup
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.DEBUG, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------------------
# ROUTES
# ----------------------------

@app.route('/')
def index():
    logging.debug("Accessed '/' route - Loading welcome page (index.html)")
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error("Error rendering index.html")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)}", 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        logging.debug("Accessed '/predict' route with GET - Loading form page (home.html)")
        try:
            return render_template('home.html')
        except Exception as e:
            logging.error("Error rendering home.html")
            logging.error(traceback.format_exc())
            return f"Error: {str(e)}", 500

    elif request.method == 'POST':
        logging.debug("Accessed '/predict' route with POST - Processing prediction request")
        try:
            # Capture form data
            data = request.form.to_dict()
            logging.debug(f"Form Data Received: {data}")

            # Create custom data object
            custom_data = CustomPredictPipeline(
                gender=data.get("gender"),
                race_ethnicity=data.get("race_ethnicity"),
                parental_level_of_education=data.get("parental_level_of_education"),
                lunch=data.get("lunch"),
                test_preparation_course=data.get("test_preparation_course"),
                reading_score=data.get("reading_score"),
                writing_score=data.get("writing_score")
            )

            # Convert to DataFrame
            df = custom_data.get_data_as_dataframe()
            logging.debug(f"DataFrame created from form input:\n{df}")

            # Prediction
            pipeline = PredictPipeline()
            prediction, model_name = pipeline.predict(df)

            logging.debug(f"Prediction Successful - Model: {model_name}, Predicted Math Score: {prediction}")

            return render_template(
                'home.html',
                results=f"{prediction:.2f}",
                model_used=model_name
            )

        except Exception as e:
            logging.error("Error occurred during prediction")
            logging.error(traceback.format_exc())
            return f"Error: {str(e)}", 500


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    logging.debug("Starting Flask application")
    app.run(host="0.0.0.0", port=5000, debug=True)


