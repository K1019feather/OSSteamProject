import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from speak import speakTrash

# Load model
MODEL_PATH = './models/waste_classification_model_20250619_215017.h5'
model = tf.keras.models.load_model(MODEL_PATH)

CATEGORIES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

def predict_frame(frame):
    processed = preprocess_frame(frame)
    preds = model.predict(processed)
    class_idx = np.argmax(preds[0])
    return CATEGORIES[class_idx]

# Camera 실행
cap = cv2.VideoCapture(0)
cv2.namedWindow("Press Space to Classify")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라를 불러올 수 없습니다.")
        break

    cv2.imshow("Press Space to Classify", frame)

    key = cv2.waitKey(1)
    if key == ord(' '):  # Spacebar 눌렀을 때
        result = predict_frame(frame)
        print(f"✅ 분류 결과: {result}")
        speakTrash(result)
    elif key == 27:  # ESC 눌렀을 때 종료
        break

cap.release()
cv2.destroyAllWindows()