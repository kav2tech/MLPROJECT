Student Performance Prediction

This project aims to predict student performance based on various academic and socio-economic features. The objective is to develop a machine learning model that can accurately estimate the **Math score** of students using other available features such as reading and writing scores, parental education, lunch type, and test preparation status.

---

## 🧠 Problem Statement

Educational institutions often seek to identify students at risk of underperforming. By predicting student scores, schools can intervene early and personalize learning plans. This project builds a machine learning pipeline to forecast student math scores, helping educators make data-driven decisions.

---

## 📂 Project Structure

MLPROJECT/
├── artifacts/                # Stores trained models and preprocessing artifacts
├── catboost_info/           # Model-specific logs/info
├── logs/                    # Project log files
├── Notebook/                # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── components/          # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline/            # Orchestration scripts
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   ├── utils.py             # Utility functions
│   ├── Exception.py         # Custom error handling
│   ├── Custom_logger.py     # Logger utility
├── Templates/               # HTML templates for web UI
│   ├── Home.html
│   ├── index.html
├── app.py                   # Flask app to serve model as web API
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation

---

## 📊 Features Used

- **Numerical:**
  - Reading Score
  - Writing Score

- **Categorical:**
  - Gender
  - Race/Ethnicity
  - Parental Level of Education
  - Lunch
  - Test Preparation Course

---

## 🔧 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Flask (Web API)
- HTML + Bootstrap (UI)
- Git for version control
- VS Code (Development IDE)

---

## 🔍 How It Works

1. **Data Ingestion:** Reads and splits data into training and testing sets.
2. **Data Transformation:** Handles missing values, encodes categorical variables, and scales features.
3. **Model Training:** Trains and evaluates regression models (e.g., Linear Regression, CatBoost).
4. **Pipeline Execution:** Training and prediction pipelines automate the full process.
5. **Flask Web App:** Exposes a user interface and a `/predict` endpoint for API access.

---

## 🚀 How to Run

### 1. Clone the Repository
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction

### 2. Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate  # For Windows
# or
source venv/bin/activate  # For Linux/Mac

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Train the Model
python src/pipeline/train_pipeline.py

### 5. Run the Flask Application
python app.py

### 6. Access the Web Interface
Open your browser and go to:  
http://127.0.0.1:5000/

---

## 🧪 API Endpoint (Postman or CURL)

- **POST** `/api/predict`  
  **Payload Example:**
  {
    "gender": "female",
    "race_ethnicity": "group B",
    "parental_level_of_education": "bachelor's degree",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 90,
    "writing_score": 88
  }

- **Response:**
  {
    "predicted_math_score": 85.72
  }

---

## ✅ To-Do / Future Enhancements

- Implement advanced models (XGBoost, LightGBM)
- Improve UI/UX with form validation and error handling
- Add Docker containerization
- Integrate with cloud (AWS/GCP) for deployment
- Add model explainability (SHAP/ELI5)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🙌 Acknowledgements

- Scikit-learn
- Flask
- Kaggle Student Performance Dataset


