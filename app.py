from flask import Flask, render_template, request  # นำเข้า Flask, render_template (สำหรับ render ไฟล์ HTML), request (สำหรับรับข้อมูลจากฟอร์ม)
import pickle  # นำเข้า pickle เพื่อใช้โหลดโมเดลที่บันทึกไว้
import pandas as pd  # นำเข้า pandas เพื่อใช้ในการจัดการข้อมูลและแปลงข้อมูลเป็น DataFrame

app = Flask(__name__)  # สร้างแอป Flask

# โหลดโมเดลที่บันทึกไว้จากไฟล์ 'model.pkl' (โหมด 'rb' คือ read binary)
model = pickle.load(open('model.pkl', 'rb'))

# โหลดข้อมูล Dataset จากไฟล์ CSV 'diabetes.csv' โดยใช้ pandas
data = pd.read_csv('diabetes.csv')

# Route สำหรับหน้าแรก
@app.route('/')
def home():
    # แสดงหน้า 'index.html' (หน้าแรกของแอปพลิเคชัน)
    return render_template('index.html')

# Route สำหรับการทำนายผลเมื่อผู้ใช้กรอกข้อมูลในฟอร์มแล้วกด submit
@app.route('/predict', methods=['POST'])
def predict():
    # รับค่าจากฟอร์มที่ผู้ใช้กรอกในหน้า 'index.html' โดยใช้ request.form.get()
    pregnancies = request.form.get('pregnancies')  # จำนวนการตั้งครรภ์
    glucose = request.form.get('glucose')  # ระดับน้ำตาลในเลือด
    blood_pressure = request.form.get('blood_pressure')  # ความดันโลหิต
    skin_thickness = request.form.get('skin_thickness')  # ความหนาของผิวหนัง
    insulin = request.form.get('insulin')  # ระดับอินซูลิน
    bmi = request.form.get('bmi')  # ดัชนีมวลกาย (BMI)
    diabetes_pedigree_function = request.form.get('diabetes_pedigree_function')  # ค่าฟังก์ชันสืบทอดทางพันธุกรรมของเบาหวาน
    age = request.form.get('age')  # อายุของผู้ใช้

    # ตรวจสอบว่าผู้ใช้กรอกข้อมูลครบทุกช่องหรือไม่
    if not (pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and diabetes_pedigree_function and age):
        # ถ้ากรอกไม่ครบ ส่งข้อความแจ้งข้อผิดพลาดและแสดงหน้าเดิม
        return render_template('index.html', prediction_text="Error: Please fill all fields")

    # สร้าง DataFrame สำหรับการทำนายจากข้อมูลที่ผู้ใช้กรอก
    # ข้อมูลจะถูกสร้างเป็น DataFrame 1 แถว โดยมีคอลัมน์ตามลำดับที่กำหนด
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # ใช้โมเดลที่โหลดไว้เพื่อทำการทำนายจากข้อมูลที่ผู้ใช้กรอก
    prediction = model.predict(input_data)[0]  # ดึงผลลัพธ์การทำนาย (0 หรือ 1)

    # แปลงผลการทำนายเป็นข้อความ (1 = Diabetic, 0 = Non-Diabetic)
    result = 'Diabetic' if prediction == 1 else 'Non-Diabetic'

    # แสดงผลลัพธ์การทำนายบนหน้า 'index.html'
    return render_template('index.html', prediction_text=f'Predicted Result: {result}')

# Route สำหรับแสดงข้อมูล Dataset
@app.route('/dataset')
def dataset():
    # แปลงข้อมูล 100 แถวแรกของ Dataset เป็น HTML โดยใช้ pandas' to_html() และส่งข้อมูลไปแสดงบนหน้า 'dataset.html'
    data_html = data.head(100).to_html(classes='table table-striped', index=False)  # กำหนดให้ไม่แสดง index
    return render_template('dataset.html', data_table=data_html)

# รันแอปพลิเคชัน Flask เมื่อเรียกใช้ไฟล์นี้โดยตรง
if __name__ == '__main__':
    app.run(debug=True)  # รันแอปในโหมด debug เพื่อให้สามารถตรวจสอบข้อผิดพลาดได้ง่าย
