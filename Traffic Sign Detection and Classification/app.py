from flask import Flask, request, render_template
import os
import uuid
import cv2
import numpy as np
import torch
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

yolo = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
cnn = load_model('cnn_model_gtsrb.h5')

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', result_img=None)

    file = request.files['image']
    img_id = str(uuid.uuid4()) + '.jpg'
    img_path = os.path.join(UPLOAD_FOLDER, img_id)
    file.save(img_path)

    results = yolo(img_path)
    detections = results.pandas().xyxy[0]

    img = cv2.imread(img_path)

    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        crop = img[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            continue

        pred = cnn.predict(preprocess(crop))
        class_id = np.argmax(pred)
        label = f"Class {class_id}"

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    result_img_path = os.path.join(UPLOAD_FOLDER, 'result_' + img_id)
    cv2.imwrite(result_img_path, img)

    return render_template('index.html', result_img=result_img_path)

if __name__ == '__main__':
    app.run(debug=True)