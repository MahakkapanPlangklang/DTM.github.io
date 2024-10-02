# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# โหลดโมเดลและ Label Encoders ที่บันทึกไว้
model = pickle.load(open('model.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    age = request.form['age']
    workclass = request.form['workclass']
    education = request.form['education']
    hours_per_week = request.form['hours_per_week']
    native_country = request.form['native_country']

    # ใช้ LabelEncoder เพื่อแปลงข้อมูลข้อความเป็นตัวเลข
    workclass = label_encoders['workclass'].transform([workclass])[0]
    education = label_encoders['education'].transform([education])[0]
    native_country = label_encoders['native_country'].transform([native_country])[0]

    # สร้าง DataFrame สำหรับการทำนาย
    input_data = pd.DataFrame([[age, workclass, education, hours_per_week, native_country]],
                              columns=['age', 'workclass', 'education', 'hours_per_week', 'native_country'])

    # ทำการทำนาย
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f'Predicted Income: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
