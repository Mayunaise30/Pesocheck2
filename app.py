from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from torchvision import transforms
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn.functional import softmax
from PIL import Image
import os
import datetime
from werkzeug.utils import secure_filename
import csv
import io
from flask import send_file

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

history = []

# Load the model with 3 classes (real, fake, invalid)
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # 3 classes: real, fake, invalid
model.load_state_dict(torch.load('best_resnet18.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def analyze_image(image_path):
    """Process an image and return prediction results"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        output = model(image_tensor)

        # Get probabilities and predicted class
        probabilities = softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        # Map the predicted index to a label (based on training: fake=0, invalid=1, real=2)
        if predicted.item() == 0:
            label = 'Counterfeit Money'
        elif predicted.item() == 1:
            label = 'Invalid (Not a Banknote)'
        else:  # predicted.item() == 2
            label = 'Real Money'
            
        confidence_score = confidence.item() * 100
        return {
            'label': label,
            'confidence': confidence_score
        }
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return {
            'label': 'Error',
            'confidence': 0
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        result = analyze_image(filepath)
        
        timestamp = datetime.datetime.now().strftime('%I:%M %p %m/%d/%Y')

        # Add to history
        history.insert(0, {
            'filename': filename,
            'label': result['label'],
            'confidence': round(result['confidence'], 2),
            'timestamp': timestamp
        })

        return render_template('index.html', filename=filename, label=result['label'], confidence=result['confidence'])

    return render_template('index.html')

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result = analyze_image(filepath)
        timestamp = datetime.datetime.now().strftime('%I:%M %p %m/%d/%Y')
        
        # Add to history
        history.insert(0, {
            'filename': filename,
            'label': result['label'],
            'confidence': round(result['confidence'], 2),
            'timestamp': timestamp
        })
        
        # Add to batch results
        results.append({
            'filename': filename,
            'label': result['label'],
            'confidence': round(result['confidence'], 2),
            'image_url': url_for('uploaded_file', filename=filename)
        })
    
    return jsonify({'results': results})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history_page():
    return render_template('history.html', history=history)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/download_history')
def download_history():
    # Create a string buffer for the CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['filename', 'label', 'confidence', 'timestamp'])
    
    # Write CSV header
    writer.writeheader()
    
    # Write history rows
    for entry in history:
        writer.writerow(entry)
    
    # Prepare the CSV for download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'pesocheck_history_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)