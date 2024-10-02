import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('diabetes.csv')

# สร้างคอลัมน์ Outcome โดยใช้เงื่อนไขจาก glyhb (เป็นเบาหวานถ้า glyhb > 6.5)
data['Outcome'] = data['glyhb'].apply(lambda x: 1 if x > 6.5 else 0)

# เลือก 5 ฟีเจอร์ที่ต้องการใช้ในการทำนาย
X = data[['chol', 'stab.glu', 'hdl', 'glyhb', 'age']]
y = data['Outcome']

# แยกข้อมูลฝึกและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดล Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# บันทึกโมเดล
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully.")
