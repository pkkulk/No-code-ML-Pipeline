# No-Code ML Pipeline Builder

This project is a simple, no-code machine learning pipeline builder built using Streamlit.
It allows users to create and run a basic ML workflow without writing any code.

## Features
- Upload CSV or Excel dataset
- View dataset information
- Apply basic preprocessing (StandardScaler / MinMaxScaler)
- Perform train-test split
- Train a classification model (Logistic Regression or Decision Tree)
- View model accuracy and confusion matrix

## Design Approach
The application follows a step-by-step pipeline similar to no-code tools like Orange.
Each step is shown only after the previous one is completed, making it beginner-friendly.

## Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
