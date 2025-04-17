import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from tensorflow.keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename

# Import GradCAM++ functions
from gradcam_utils import compute_gradcam_plus_plus, overlay_gradcam

app = Flask(__name__)
app.secret_key = "pulmonova_secret"

# Use /tmp/ for temporary storage (Render requirement)
UPLOAD_FOLDER = '/tmp/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = os.path.join(app.root_path, 'models', 'my_model.h5')
model = load_model(MODEL_PATH)

# Model configuration
IMG_SIZE = 150
LAST_CONV_LAYER_NAME = 'conv2d_4'  # Change if needed

# Utility: Preprocess input X-ray image
def preprocess_image(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, None
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_4d = np.reshape(img_normalized, (1, IMG_SIZE, IMG_SIZE, 1))
    return img_resized, img_4d

# Predict and generate GradCAM++ heatmap
def make_prediction_and_gradcam(img_4d):
    prediction = model.predict(img_4d)[0][0]
    if prediction > 0.5:
        pred_class = 'NORMAL'
        confidence = prediction * 100
    else:
        pred_class = 'PNEUMONIA'
        confidence = (1 - prediction) * 100

    heatmap = compute_gradcam_plus_plus(model, img_4d, layer_name=LAST_CONV_LAYER_NAME)
    return pred_class, confidence, heatmap

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.')
            return redirect(url_for('index'))

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.')
            return redirect(url_for('index'))

        # Retrieve patient info
        patient_id = request.form.get("patient_id")
        age = request.form.get("age")
        gender = request.form.get("gender")
        symptoms = request.form.get("symptoms")

        if not all([patient_id, age, gender, symptoms]):
            flash("Please fill in all patient information before uploading an image.")
            return redirect(url_for('index'))

        patient_id_filename = secure_filename(patient_id)
        original_filename = secure_filename(file.filename)
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # Preprocess
        img_resized, img_4d = preprocess_image(original_path)
        if img_resized is None or img_4d is None:
            flash('Failed to process image. Please upload a valid image.')
            return redirect(url_for('index'))

        # Predict and generate GradCAM++
        pred_class, confidence, heatmap = make_prediction_and_gradcam(img_4d)
        overlay_img = overlay_gradcam(heatmap, img_resized, alpha=0.4)

        # Save files with unique names
        original_saved = f"original_{patient_id_filename}.jpg"
        gradcam_saved = f"gradcam_{patient_id_filename}.jpg"
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], original_saved), img_resized)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], gradcam_saved), overlay_img)

        return render_template('result.html',
                               original_img_url=url_for('uploaded_file', filename=original_saved),
                               gradcam_img_url=url_for('uploaded_file', filename=gradcam_saved),
                               prediction=pred_class,
                               confidence=round(confidence, 2),
                               patient_id=patient_id,
                               age=age,
                               gender=gender,
                               symptoms=symptoms)

    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
