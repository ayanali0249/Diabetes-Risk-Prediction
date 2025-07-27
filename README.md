# Early Diabetes Risk Prediction – ML Project

This project presents a machine learning solution for predicting early diabetes risk using medical data. It was completed as part of a technical task for the AI/ML Developer Role at Rx.Now. The solution includes data preprocessing, model development, SHAP-based explainability, and an interactive Streamlit interface for real-time predictions.

## Project Overview

The goal of this project is to build a prototype that demonstrates early disease risk prediction capabilities using publicly available data. The PIMA Indians Diabetes Dataset is used for building a classification model to identify patients at risk of diabetes. The entire pipeline includes:

- Data preprocessing and cleaning
- Model building using Random Forest
- Model evaluation using classification metrics
- Explainability using SHAP
- Interactive prediction interface using Streamlit

## Tech Stack and Tools

Programming Languages: Python (Pandas, Scikit-learn, Matplotlib, Seaborn, SHAP, Streamlit)  
Deployment: Streamlit  
Model Persistence: Pickle  
IDE/Platform: Google Colab, Jupyter Notebook, VS Code  

## Key Features

Data Preprocessing:
- Replacement of zero values in medical fields with NaN
- Median imputation for missing values
- StandardScaler normalization of numerical features

Model Building:
- Random Forest Classifier trained on normalized features
- Evaluation using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix

Model Explainability:
- SHAP summary plot to interpret feature impact on predictions

Interactive Web App:
- Developed using Streamlit
- Accepts user inputs for medical data
- Displays risk prediction results in real-time

## Visualizations

- Feature importance bar chart for Random Forest model
- SHAP summary plot for top features
- Confusion matrix and classification report

## How to Use

Clone the repository:
git clone https://github.com/<your-username>/diabetes-risk-predictor.git  
cd diabetes-risk-predictor  

Install required Python libraries:
pip install pandas numpy matplotlib seaborn scikit-learn shap streamlit pickle-mixin  

Run the Jupyter Notebook:
Open `diabetes_prediction.ipynb` in Google Colab or Jupyter Notebook  
Run all cells in order to train and evaluate the model  
Generate and save the trained model and scaler as `diabetes_model.pkl` and `scaler.pkl`  

Run the Streamlit App:
Ensure `app.py`, `diabetes_model.pkl`, and `scaler.pkl` are in the same directory  
Execute the following command in terminal:
streamlit run app.py  

## Folder Structure

├── diabetes_prediction.ipynb       # Jupyter notebook with complete ML workflow  
├── diabetes_model.pkl              # Trained Random Forest model  
├── scaler.pkl                      # Scaler used for normalizing user inputs  
├── app.py                          # Streamlit app for prediction  
├── README.md                       # Project documentation  


## Screenshots

- **SHAP Plot** – Shows top features influencing predictions  
  <img width="440" height="680" alt="image" src="https://github.com/user-attachments/assets/707f0e77-fdd4-4185-9dd7-d42725cd546b" />

- **Streamlit App** – Interactive interface for diabetes risk prediction  
  <img width="1920" height="965" alt="Screenshot 2025-07-27 145858" src="https://github.com/user-attachments/assets/5537480f-6c19-4526-8f90-3605e7b8bff2" />
  <img width="1920" height="953" alt="Screenshot 2025-07-27 145813" src="https://github.com/user-attachments/assets/fdb82075-7a14-4b21-b031-cd9ce6201b11" />

- **Feature Importance** – Visualizes key factors from the Random Forest model  
  <img width="706" height="435" alt="image" src="https://github.com/user-attachments/assets/5af40ab9-32c8-4764-b96b-d9c4110500bd" />


## Contact

For any queries, feel free to reach out:

Ayan Ali  
Email: ayanali0249@gmail.com  
LinkedIn: https://linkedin.com/in/ayan-ali0249  
Instagram: https://instagram.com/ayan_ali_0249
