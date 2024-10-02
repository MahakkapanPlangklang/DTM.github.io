import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# โหลดข้อมูลจากไฟล์ CSV ที่มีฟีเจอร์ใหม่
data = pd.read_csv('diabetes.csv')

# เลือกฟีเจอร์ที่ต้องการใช้ในการทำนาย
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
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
