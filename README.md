# Loan Approval Prediction Web App

A beginner-friendly, fully local Flask application that serves a trained machine learning model for loan approval prediction. The app accepts applicant details through a responsive web form, applies the same preprocessing used during training, and returns an approval decision with a confidence score.

## Internship Details

| Field | Details |
| --- | --- |
| Intern ID | CITS6708 |
| Full Name | Aditya Kiratsata |
| No. of Weeks | 4 |
| Project Name | Loan Approval Predictor |
| Project Scope | Intermediate |

## Project Overview

This project demonstrates how to deploy a trained scikit-learn model behind a Flask web interface. It is designed to run entirely on a local machine and does not require any external services or cloud deployment.

The application uses three serialized artifacts stored in the project root:

- `loan_approval_model.pkl` - trained Random Forest classifier
- `feature_names.pkl` - exact feature order used by the model
- `feature_mappings.pkl` - categorical value mappings used during preprocessing

## Features

- Local Flask web application with a clean Bootstrap 5 UI
- Safe startup loading of model artifacts
- Input validation for all user-facing fields
- Feature mapping and feature engineering consistent with training
- Log transforms for numeric stability
- Prediction result page with confidence score
- Validation errors shown clearly without crashing the server

## Tech Stack

- Python 3
- Flask
- pandas
- numpy
- joblib
- scikit-learn
- Bootstrap 5

## Repository Structure

```text
Loan-Approval-Predictor/
├── app.py
├── feature_mappings.pkl
├── feature_names.pkl
├── loan-approval-predictor.ipynb
├── loan_approval_model.pkl
├── requirements.txt
└── templates/
    ├── index.html
    └── prediction.html
```

## Prerequisites

- Python 3.10+ recommended
- A local terminal or command prompt
- The three pickle files present in the project root

## Installation

1. Open the project folder in VS Code or your preferred editor.
2. Create and activate a virtual environment.
3. Install the dependencies.

```bash
pip install -r requirements.txt
```

## How to Run

Start the Flask app from the project root:

```bash
python app.py
```

Then open the application in your browser:

```text
http://127.0.0.1:5000
```

## How It Works

1. The user enters applicant details in the form.
2. The backend validates the inputs before prediction.
3. Categorical values are converted using `feature_mappings.pkl`.
4. Numeric features are transformed to match training-time preprocessing.
5. The model predicts loan approval or rejection.
6. A confidence score and application summary are shown on the result page.

## Input Validation Rules

The app checks the following before calling the model:

- `ApplicantIncome` must be a non-negative number
- `CoapplicantIncome` must be a non-negative number
- `LoanAmount` must be within a reasonable range
- `Loan_Amount_Term` must be between 12 and 480 months
- `Credit_History` must be `0.0` or `1.0`
- Dropdown fields must use allowed values only

This prevents invalid values such as an unrealistic loan term from reaching the model.

## Feature Engineering

The following derived features are created before inference:

- `Total_Income = ApplicantIncome + CoapplicantIncome`
- `Total_Income_Log = log1p(Total_Income)`
- `LoanAmount_Log = log1p(LoanAmount)`

The final inference dataframe is reordered to exactly match the model's stored feature list.

## Default Feature Order

The model expects the following feature order:

1. `Gender`
2. `Married`
3. `Dependents`
4. `Education`
5. `Self_Employed`
6. `Credit_History`
7. `Property_Area`
8. `Loan_Amount_Term`
9. `LoanAmount_Log`
10. `Total_Income_Log`

## Troubleshooting

### Model artifact loading warning

If you see a scikit-learn compatibility warning during startup, it usually means the local installed version of scikit-learn differs from the version used when the pickle file was created. The safest fix is to install the exact dependencies listed in `requirements.txt`.

### Prediction errors

If the result page shows a validation error, check the values entered in the form. The app is intentionally strict about data quality so incorrect values are not passed to the model.

### Missing pickle files

If the app cannot start, make sure these files exist in the project root:

- `loan_approval_model.pkl`
- `feature_names.pkl`
- `feature_mappings.pkl`