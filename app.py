from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# โหลดโมเดล
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    chol = request.form.get('chol', None)
    stab_glu = request.form.get('stab_glu', None)
    hdl = request.form.get('hdl', None)
    glyhb = request.form.get('glyhb', None)
    age = request.form.get('age', None)
    
    # ตรวจสอบว่ากรอกข้อมูลครบหรือไม่
    if not chol or not stab_glu or not hdl or not glyhb or not age:
        return render_template('index.html', prediction_text="Error: Please fill all fields")

    # สร้าง DataFrame สำหรับการทำนาย
    input_data = pd.DataFrame([[chol, stab_glu, hdl, glyhb, age]],
                              columns=['chol', 'stab.glu', 'hdl', 'glyhb', 'age'])

    # ทำการทำนาย
    prediction = model.predict(input_data)[0]

    # แสดงผลการพยากรณ์
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return render_template('index.html', prediction_text=f'Predicted Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)
