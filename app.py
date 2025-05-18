import os
import pickle
import numpy as np
from PIL import Image
from flask import flash

from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model 


# --- Configuration ---
BASE_DIR      = os.environ.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(BASE_DIR, 'uploads'))

# create uploads folder if needed
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# allowed extensions
ALLOWED_EXTS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Severity Configuration ---
severity_config = {
    "thresholds": [0.5, 0.7, 0.9],
    "labels":     ["Normal", "Mild Pneumonia", "Moderate Pneumonia", "Severe Pneumonia"]
}

def get_severity_label(p):
    if   p < 0.5: return "Normal"
    elif p < 0.7: return "Mild Pneumonia"
    elif p < 0.9: return "Moderate Pneumonia"
    else:         return "Severe Pneumonia"

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Models & Pipelines ---
feature_extractor = load_model(os.path.join(MODEL_DIR, 'feature_extractor.h5'), compile=False)

with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'rb') as f:
    svm_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'severity_mapping.pkl'), 'rb') as f:
    severity_mapping = pickle.load(f)


# --- Utility Functions ---
def is_allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    # DenseNet preprocess_input expects BGR ordering and mean centering,
    # but if you only trained with [0,1] scaling, skip preprocess_input
    # from tensorflow.keras.applications.densenet import preprocess_input
    # arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        if not is_allowed(file.filename):
            return render_template('index.html', error="Invalid file type.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- Process image and make prediction ---
        img_arr      = preprocess_image(filepath)
        features     = feature_extractor.predict(img_arr)
        features_scl = scaler.transform(features)

        pred_label = svm_model.predict(features_scl)[0]
        label_str  = 'PNEUMONIA' if pred_label == 1 else 'NORMAL'

        # Initialize severity values
        severity_text = None
        severity_percent = None
        confidence = None

        if label_str == 'PNEUMONIA':
            # Compute raw score for severity calculation
            raw_score = float(svm_model.decision_function(features_scl)[0])
            
            # Normalize score to a percentage (0-100) for display
            # Assuming scores typically range from -3 to +3 in SVM
            normalized_score = (raw_score + 3) / 6  # Convert to 0-1 range
            severity_percent = min(100, max(0, int(normalized_score * 100)))
            
            # Get severity text based on the normalized score using the new function
            severity_text = get_severity_label(normalized_score)
            
            # Calculate confidence dynamically based on distance from decision boundary
            # The further from 0, the more confident the model is
            confidence = min(99, max(50, int(abs(raw_score) * 20)))
            
            print(f"Raw score: {raw_score}, Normalized: {normalized_score:.2f}, Severity: {severity_text} ({severity_percent}%), Confidence: {confidence}%")
        else:
            # If normal, set appropriate values
            severity_text = "Normal"
            severity_percent = 0
            
            # For normal cases, also calculate confidence based on raw score
            raw_score = float(svm_model.decision_function(features_scl)[0])
            confidence = min(99, max(50, int(abs(raw_score) * 20)))

        return render_template(
            'results.html',
            filename=filename,
            label=label_str,
            severity_text=severity_text,
            severity_percent=severity_percent,
            confidence=confidence
        )

    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded image so it can be displayed in results."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/dashboard')
def dashboard():
    # Hard-coded model performance metrics
    metrics = {
        'accuracy':       '82.53%',
        'auc':            '0.968',  
        'precision':      '78.58%',
        'recall':         '99.23%',
        'f1':             '87.98%',
        'true_negative':  128,
        'false_positive': 106,
        'false_negative':   3,
        'true_positive':  387
    }


    return render_template('dashboard.html', metrics=metrics)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Extract submitted form fields
        prediction_id = request.form.get('prediction_id')
        is_correct    = request.form.get('is_correct')
        correct_diag  = request.form.get('correct_diagnosis')
        severity_acc  = request.form.get('severity_accuracy')
        comments      = request.form.get('comments')
        is_prof       = request.form.get('is_medical_professional') == 'on'

        # Validate required field
        if not is_correct:
            return render_template('feedback.html', error="Please select whether the prediction was correct.")

        # You can save this feedback to a CSV file, database, or log (simplest form below):
        feedback_log_path = os.path.join(BASE_DIR, 'feedback_log.txt')
        with open(feedback_log_path, 'a') as f:
            f.write(f"Prediction ID: {prediction_id}\n")
            f.write(f"Correct: {is_correct}\n")
            f.write(f"Diagnosis: {correct_diag}\n")
            f.write(f"Severity Accuracy: {severity_acc}\n")
            f.write(f"Comments: {comments}\n")
            f.write(f"Medical Professional: {is_prof}\n")
            f.write(f"{'-'*40}\n")

        return render_template('feedback.html', success=True)

    # If GET request, show empty form
    return render_template('feedback.html')


if __name__ == '__main__':
    app.run(debug=True)