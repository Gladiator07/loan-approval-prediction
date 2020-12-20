from flask import Flask,render_template,url_for,request
import pickle
import numpy as np

app=Flask(__name__)

model_path = 'Trained_Model/log_reg.pkl'
model = pickle.load(
    open(model_path, 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    # Getting the data from the form
    gender=request.form['gender']
    status=request.form['married']
    education=request.form['education']
    work=request.form['employment']
    app_income=request.form['applicant_income']
    co_app_income=request.form['coapplicant_income']
    loan_amount=request.form['loan_amount']
    loan_amount_term=request.form['loan_amount_term']
    credit_history = request.form['credit_history']
    property_area = request.form['property_area']
    dependents = request.form['dependents']

    query = np.array([[gender, status, education, work, int(app_income), 
                        float(co_app_income), float(loan_amount), float(loan_amount_term), float(credit_history), property_area, dependents]])

    prediction = model.predict(query)

    if prediction == 0:
        response = "Sorry, You are not eligible for the loan 😔😔"
    else:
        response = "Congrats, You are eligible for the loan 😄😄"

    
    return render_template('result.html', prediction=response)


if __name__ == '__main__':
    app.run(debug=True)
