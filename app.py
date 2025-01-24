from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import re
from drug_name import DrugModelPredictor
from PyPDF2 import PdfReader
from PIL import Image
from pytesseract import image_to_string

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the drug model predictor
predictor = DrugModelPredictor()

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip() or "No readable text found in the PDF."
    except Exception as e:
        return f"Error reading PDF file: {e}"

def extract_text_from_image(file_path):
    """Extract text from an image file."""
    try:
        image = Image.open(file_path)
        return image_to_string(image)
    except Exception as e:
        return f"Error processing image for OCR: {e}"

def extract_drug_names(text):
    """Extract drug names from text using improved pattern matching."""
    patterns = [
        r'\b[A-Z][a-z]+(?:[-][a-zA-Z]+)*(?:\s+\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg))?\b',
        r'\b[A-Z][a-z]+(?:[-]\d+)?\b',
        r'\b[A-Z][a-z]+(?:[-/][A-Z][a-z]+)+\b'
    ]
    
    combined_pattern = '|'.join(patterns)
    
    exclude_words = {'the', 'and', 'or', 'with', 'without', 'take', 'daily', 'dose',
                    'tablet', 'capsule', 'prescription', 'medicine', 'medication',
                    'patient', 'doctor', 'hospital', 'clinic', 'pharmacy', 'times'}
    
    potential_drugs = re.findall(combined_pattern, text)
    
    drug_names = []
    for drug in potential_drugs:
        clean_drug = re.split(r'\s+\d', drug)[0]
        clean_drug = clean_drug.title()
        
        if clean_drug.lower() not in exclude_words:
            drug_names.append(clean_drug)
    
    return list(set(drug_names))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    original_text, wrong_text = None, None

    try:
        # Handle original input
        if request.form['input_type'] == 'text':
            original_text = request.form['original_report']
        elif request.form['input_type'] == 'image':
            file = request.files['original_image']
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                original_text = extract_text_from_image(file_path)
            else:
                flash('Invalid or missing image file for the original report.', 'danger')
        elif request.form['input_type'] == 'pdf':
            file = request.files['original_pdf']
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                original_text = extract_text_from_pdf(file_path)
            else:
                flash('Invalid or missing PDF file for the original report.', 'danger')

        # Handle wrong transcription input
        if request.form['wrong_input_type'] == 'text':
            wrong_text = request.form['wrong_report']
        elif request.form['wrong_input_type'] == 'image':
            file = request.files['wrong_image']
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                wrong_text = extract_text_from_image(file_path)
            else:
                flash('Invalid or missing image file for the wrong transcription.', 'danger')
        elif request.form['wrong_input_type'] == 'pdf':
            file = request.files['wrong_pdf']
            if file and allowed_file(file.filename):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(file_path)
                wrong_text = extract_text_from_pdf(file_path)
            else:
                flash('Invalid or missing PDF file for the wrong transcription.', 'danger')

        # Validate input
        if not original_text or not wrong_text:
            flash('Both inputs (original and transcription) are required and cannot be empty.', 'danger')
            return redirect(url_for('home'))

        # Extract and compare drug names
        original_drugs = extract_drug_names(original_text)
        wrong_drugs = extract_drug_names(wrong_text)
        missing_drugs = list(set(original_drugs) - set(wrong_drugs))
        extra_drugs = list(set(wrong_drugs) - set(original_drugs))

        # Compare sentences
        original_sentences = set(map(str.strip, original_text.splitlines()))
        wrong_sentences = set(map(str.strip, wrong_text.splitlines()))
        missing_sentences = list(original_sentences - wrong_sentences)
        extra_sentences = list(wrong_sentences - original_sentences)

        # Predict condition based on drug names
        try:
            predicted_condition = predictor.predict_condition(original_drugs)
        except Exception as e:
            predicted_condition = f"Error in prediction: {str(e)}"

        # Calculate similarity percentage
        similarity_percentage = 0
        if original_drugs and wrong_drugs:
            similarity_percentage = len(set(original_drugs) & set(wrong_drugs)) / len(set(original_drugs) | set(wrong_drugs)) * 100

        return render_template('away.html',
                               original_text=original_text,
                               wrong_text=wrong_text,
                               missing_drugs=missing_drugs,
                               extra_drugs=extra_drugs,
                               missing_sentences=missing_sentences,
                               extra_sentences=extra_sentences,
                               predicted_condition=predicted_condition,
                               similarity_percentage=round(similarity_percentage, 2))
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'danger')
        return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)