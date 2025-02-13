from flask import Flask, render_template, request, redirect, url_for
import os
from PyPDF2 import PdfReader
import nltk
from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Download NLTK data
nltk.download('punkt')

# MCQ generation pipeline
mcq_pipeline = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_mcqs(text):
    sentences = nltk.sent_tokenize(text)
    mcqs = []
    for sentence in sentences:
        if len(sentence.split()) > 5:  # Filter out short sentences
            result = mcq_pipeline(sentence)
            mcqs.append(result[0]['generated_text'])
    return mcqs

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            text = extract_text_from_pdf(filepath)
            mcqs = generate_mcqs(text)
            return render_template('index.html', mcqs=mcqs)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
