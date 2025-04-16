import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename

# Import GradCAM++ functions from gradcam_utils.py
from gradcam_utils import compute_gradcam_plus_plus, overlay_gradcam

app = Flask(__name__)
app.secret_key = "pulmonova_secret"

# Configure upload folder (static/uploads)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model (adjust MODEL_PATH to point to your model file)
MODEL_PATH = r'C:\pneumonia-detection\my_model.keras'
model = load_model(MODEL_PATH)

IMG_SIZE = 150
LAST_CONV_LAYER_NAME = 'conv2d_4'  # Replace with your model's last conv layer name

def preprocess_image(img_path):
    """
    Loads the image in grayscale, resizes it, and normalizes pixel values.
    Returns:
      - img_resized: image for display,
      - img_4d: 4D array for model inference.
    """
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, None
    
    # Resize for model input
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_4d = np.reshape(img_normalized, (1, IMG_SIZE, IMG_SIZE, 1))
    return img_resized, img_4d

def make_prediction_and_gradcam(img_4d):
    """
    Predicts using the model and generates the GradCAM++ heatmap.
    Returns: predicted class, confidence, and the heatmap.
    """
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
        
        # Retrieve patient info fields
        patient_id = request.form.get("patient_id")
        age = request.form.get("age")
        gender = request.form.get("gender")
        symptoms = request.form.get("symptoms")
        if not all([patient_id, age, gender, symptoms]):
            flash("Please fill in all patient information before uploading an image.")
            return redirect(url_for('index'))
        
        # Secure the patient id for filename use
        patient_id_filename = secure_filename(patient_id)
        
        # Save the uploaded file (this is optional if you don't need the raw file)
        original_file = file
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(original_file.filename))
        original_file.save(original_filepath)

        # Preprocess the image for model inference
        img_resized, img_4d = preprocess_image(original_filepath)
        if img_resized is None or img_4d is None:
            flash('Failed to process image. Please upload a valid image.')
            return redirect(url_for('index'))
        
        # Prediction and GradCAM computation
        pred_class, confidence, heatmap = make_prediction_and_gradcam(img_4d)
        overlay_img = overlay_gradcam(heatmap, img_resized, alpha=0.4)
        
        # Build new filenames using patient id
        original_filename = f"original_{patient_id_filename}.jpg"
        gradcam_filename = f"gradcam_{patient_id_filename}.jpg"
        display_original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        
        cv2.imwrite(display_original_path, img_resized)
        cv2.imwrite(gradcam_path, overlay_img)
        
        return render_template('result.html',
                               original_img_url='uploads/' + original_filename,
                               gradcam_img_url='uploads/' + gradcam_filename,
                               prediction=pred_class,
                               confidence=confidence,
                               patient_id=patient_id,
                               age=age,
                               gender=gender,
                               symptoms=symptoms)
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
