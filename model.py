import pandas as pd  # นำเข้า pandas สำหรับการจัดการข้อมูล
from sklearn.model_selection import train_test_split  # นำเข้า train_test_split เพื่อใช้แบ่งข้อมูลสำหรับการฝึกและทดสอบ
from sklearn.tree import DecisionTreeClassifier  # นำเข้า DecisionTreeClassifier สำหรับสร้างโมเดล Decision Tree
import pickle  # นำเข้า pickle เพื่อใช้บันทึกโมเดลลงไฟล์

# โหลดข้อมูลจากไฟล์ CSV 'diabetes.csv' โดยใช้ pandas
# ข้อมูลนี้เป็น dataset ที่ใช้ในการทำนายเบาหวาน
data = pd.read_csv('diabetes.csv')

# เลือกฟีเจอร์ (features) ที่ต้องการใช้ในการทำนาย (X)
# ฟีเจอร์เหล่านี้ได้แก่ Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction และ Age
# ฟีเจอร์เหล่านี้เป็นตัวแปรอิสระที่ใช้ในการฝึกโมเดล
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

# เลือกเป้าหมาย (target) ที่ต้องการทำนาย (y)
# 'Outcome' เป็นคอลัมน์ที่บอกว่าผู้ป่วยเป็นเบาหวานหรือไม่ (1 = Diabetic, 0 = Non-Diabetic)
y = data['Outcome']

# แยกข้อมูลออกเป็นชุดข้อมูลฝึก (training set) และชุดข้อมูลทดสอบ (testing set)
# ใช้ฟังก์ชัน train_test_split จาก sklearn เพื่อแบ่งข้อมูลเป็นสองส่วน:
# - 70% ของข้อมูลจะใช้สำหรับการฝึก (X_train, y_train)
# - 30% ของข้อมูลจะใช้สำหรับการทดสอบ (X_test, y_test)
# random_state=42 กำหนดให้การสุ่มข้อมูลเป็นแบบเดียวกันทุกครั้ง เพื่อให้ผลลัพธ์สามารถทำซ้ำได้
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดล Decision Tree โดยใช้ DecisionTreeClassifier จาก sklearn
# โมเดลนี้จะเรียนรู้จากข้อมูลฝึก (X_train, y_train)
model = DecisionTreeClassifier()

# ฝึกโมเดลโดยใช้ข้อมูลฝึก (X_train, y_train)
# คำสั่งนี้จะทำให้โมเดลเรียนรู้รูปแบบจากข้อมูลเพื่อใช้ในการทำนาย
model.fit(X_train, y_train)

# บันทึกโมเดลที่ฝึกเสร็จแล้วลงไฟล์ 'model.pkl'
# ใช้ pickle ในการเขียนโมเดลลงไฟล์ โดยเปิดไฟล์ในโหมด 'wb' (write binary)
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)  # ใช้ pickle.dump เพื่อบันทึกโมเดลลงในไฟล์

# แสดงข้อความเพื่อบอกว่าโมเดลถูกบันทึกเรียบร้อยแล้ว
print("Model saved successfully.")
