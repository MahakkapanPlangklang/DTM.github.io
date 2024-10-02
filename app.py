from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    pregnancies = request.form.get('pregnancies')
    glucose = request.form.get('glucose')
    blood_pressure = request.form.get('blood_pressure')
    skin_thickness = request.form.get('skin_thickness')
    insulin = request.form.get('insulin')
    bmi = request.form.get('bmi')
    diabetes_pedigree_function = request.form.get('diabetes_pedigree_function')
    age = request.form.get('age')

    # ตรวจสอบว่ากรอกข้อมูลครบหรือไม่
    if not (pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and diabetes_pedigree_function and age):
        return render_template('index.html', prediction_text="Error: Please fill all fields")

    # สร้าง DataFrame สำหรับการทำนาย
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # ทำการทำนาย
    prediction = model.predict(input_data)[0]

    # แสดงผลการพยากรณ์
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return render_template('index.html', prediction_text=f'Predicted Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
