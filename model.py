# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv('income_data.csv')

# แปลงข้อมูลข้อความเป็นตัวเลข
label_encoders = {}
for column in ['workclass', 'education', 'native_country']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # เก็บ LabelEncoder เพื่อใช้ทำนายภายหลัง

# เลือก features และ target
X = data[['age', 'workclass', 'education', 'hours_per_week', 'native_country']]  # features
y = data['income']  # target

# แยกข้อมูลสำหรับการฝึกฝนและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# สร้างโมเดล Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# บันทึกโมเดล
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# บันทึก LabelEncoders เพื่อใช้ในภายหลัง
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
