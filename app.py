from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import re
from drug_name import DrugModelPredictor
from PyPDF2 import PdfReader
from PIL import Image
from pytesseract import image_to_string, pytesseract

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the path to the Tesseract executable (adjust as needed for your environment)
pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')

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
    except FileNotFoundError:
        return "Error: Image file not found."
    except OSError:
        return "Error: Invalid image file format."
    except Exception as e:
        return f"Error processing image for OCR: {e}. Make sure Tesseract is installed and its path is correctly configured."

def extract_drug_names(text):
    """Extract drug names from text using improved pattern matching."""
    return predictor.predict(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Handle different file types
            if filename.rsplit('.', 1)[1].lower() == 'pdf':
                extracted_text = extract_text_from_pdf(file_path)
            else:
                extracted_text = extract_text_from_image(file_path)

            drug_names = extract_drug_names(extracted_text)
            return render_template('result.html', drugs=drug_names)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
