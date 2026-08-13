from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import traceback

app = Flask(__name__)

ALLOWED_VALUES = {
    'Gender': {'Male', 'Female'},
    'Married': {'Yes', 'No'},
    'Dependents': {'0', '1', '2', '3+'},
    'Education': {'Graduate', 'Not Graduate'},
    'Self_Employed': {'Yes', 'No'},
    'Property_Area': {'Rural', 'Semiurban', 'Urban'},
    'Credit_History': {'0.0', '1.0', 0.0, 1.0},
}

NUMERIC_RANGES = {
    'ApplicantIncome': (0.0, 10000000.0),
    'CoapplicantIncome': (0.0, 10000000.0),
    'LoanAmount': (0.0, 10000.0),
    'Loan_Amount_Term': (12.0, 480.0),
}

# Load the model and artifacts at startup
try:
    model = joblib.load('loan_approval_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    feature_mappings = joblib.load('feature_mappings.pkl')
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model or artifacts: {e}")
    traceback.print_exc()
    # In a production setting, you might want to exit or handle this more gracefully
    # For now, we'll let the app start but routes will fail if model is not loaded.
    model = None
    feature_names = None
    feature_mappings = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def _parse_choice(field_name, raw_value):
    if raw_value is None or raw_value == '':
        raise ValueError(f'{field_name} is required.')

    if field_name == 'Credit_History':
        try:
            parsed_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError('Credit History must be either 0.0 or 1.0.') from exc

        if parsed_value not in (0.0, 1.0):
            raise ValueError('Credit History must be either 0.0 or 1.0.')

        return parsed_value

    if raw_value not in ALLOWED_VALUES[field_name]:
        raise ValueError(f'Invalid value for {field_name}.')

    return raw_value


def _parse_numeric(field_name, raw_value):
    if raw_value is None or raw_value == '':
        raise ValueError(f'{field_name} is required.')

    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field_name} must be a valid number.') from exc

    minimum, maximum = NUMERIC_RANGES[field_name]
    if parsed_value < minimum or parsed_value > maximum:
        if field_name == 'Loan_Amount_Term':
            raise ValueError('Loan Amount Term must be between 12 and 480 months.')
        if field_name == 'LoanAmount':
            raise ValueError('Loan Amount must be between 0 and 10000 thousand.')
        raise ValueError(f'{field_name} must be between {minimum} and {maximum}.')

    return parsed_value

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or feature_names is None or feature_mappings is None:
        return render_template('prediction.html', 
                               error="Model not loaded. Please check server logs.",
                               status=None, 
                               confidence=None, 
                               inputs=None)
    
    try:
        # Extract and validate form data.
        gender = _parse_choice('Gender', request.form.get('Gender'))
        married = _parse_choice('Married', request.form.get('Married'))
        dependents = _parse_choice('Dependents', request.form.get('Dependents'))
        education = _parse_choice('Education', request.form.get('Education'))
        self_employed = _parse_choice('Self_Employed', request.form.get('Self_Employed'))
        applicant_income = _parse_numeric('ApplicantIncome', request.form.get('ApplicantIncome'))
        coapplicant_income = _parse_numeric('CoapplicantIncome', request.form.get('CoapplicantIncome'))
        loan_amount = _parse_numeric('LoanAmount', request.form.get('LoanAmount'))
        loan_amount_term = _parse_numeric('Loan_Amount_Term', request.form.get('Loan_Amount_Term'))
        credit_history = _parse_choice('Credit_History', request.form.get('Credit_History'))
        property_area = _parse_choice('Property_Area', request.form.get('Property_Area'))

        # Map categorical features using the provided mappings
        gender_mapped = feature_mappings['Gender'][gender]
        married_mapped = feature_mappings['Married'][married]
        dependents_mapped = feature_mappings['Dependents'][dependents]
        education_mapped = feature_mappings['Education'][education]
        self_employed_mapped = feature_mappings['Self_Employed'][self_employed]
        property_area_mapped = feature_mappings['Property_Area'][property_area]
        # Credit history is already a float, no mapping needed

        # Feature engineering
        total_income = applicant_income + coapplicant_income
        if total_income < 0:
            raise ValueError('Total income cannot be negative.')
        total_income_log = np.log1p(total_income)
        loan_amount_log = np.log1p(loan_amount)

        # Create a DataFrame with the correct feature order
        input_data = {
            'Gender': gender_mapped,
            'Married': married_mapped,
            'Dependents': dependents_mapped,
            'Education': education_mapped,
            'Self_Employed': self_employed_mapped,
            'Credit_History': credit_history,
            'Property_Area': property_area_mapped,
            'Loan_Amount_Term': loan_amount_term,
            'LoanAmount_Log': loan_amount_log,
            'Total_Income_Log': total_income_log
        }
        input_df = pd.DataFrame([input_data])
        # Ensure the columns are in the exact order as in feature_names
        input_df = input_df[feature_names]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        # Get the confidence for the predicted class
        confidence = probabilities[prediction] * 100  # as percentage

        # Map prediction to status and theme
        if prediction == 1:
            status = "Approved"
            theme = "success"
        else:
            status = "Rejected"
            theme = "danger"

        # Prepare inputs for display (original values)
        inputs = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': int(applicant_income) if applicant_income.is_integer() else applicant_income,
            'CoapplicantIncome': int(coapplicant_income) if coapplicant_income.is_integer() else coapplicant_income,
            'LoanAmount': int(loan_amount) if loan_amount.is_integer() else loan_amount,
            'Loan_Amount_Term': int(loan_amount_term) if loan_amount_term.is_integer() else loan_amount_term,
            'Credit_History': f'{credit_history:.1f}',
            'Property_Area': property_area
        }

        return render_template('prediction.html', 
                               status=status, 
                               theme=theme,
                               confidence=f'{confidence:.2f}',
                               inputs=inputs)

    except Exception as e:
        # Log the error for debugging
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return render_template('prediction.html', 
                               error=str(e),
                               status=None, 
                               confidence=None, 
                               inputs=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)