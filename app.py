# Import required libraries for Flask web app, image processing, and machine learning
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, jsonify
from torchvision import transforms  # For image transformations
import torch  # PyTorch for model operations
import torch.nn as nn  # Neural network modules
from torchvision.models import resnet18  # Pre-trained ResNet18 model
from torch.nn.functional import softmax  # Softmax for probability outputs
from PIL import Image  # Image handling
import os  # File system operations
import datetime  # Timestamp generation
from werkzeug.utils import secure_filename  # Secure file uploads
import csv  # CSV file handling
import io  # In-memory file operations
from flask import send_file  # File download support

# Initialize Flask application
app = Flask(__name__)

# Configure upload folder for storing images
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# List to store analysis history
history = []

# Load and configure the ResNet18 model for 3-class classification
model = resnet18(pretrained=False)  # Initialize ResNet18 without pre-trained weights
num_ftrs = model.fc.in_features  # Get number of input features for final layer
model.fc = nn.Linear(num_ftrs, 3)  # Modify final layer for 3 classes: real, fake, invalid
model.load_state_dict(torch.load('best_resnet18.pth', map_location=torch.device('cpu')))  # Load trained weights
model.eval()  # Set model to evaluation mode

# Define image transformation pipeline for preprocessing
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize image to 640x640
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

def analyze_image(image_path):
    """Process an image and return prediction results"""
    try:
        # Open and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        # Apply transformations and add batch dimension
        image_tensor = transform(image).unsqueeze(0)
        # Get model predictions
        output = model(image_tensor)

        # Compute probabilities and get predicted class
        probabilities = softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        # Map predicted index to label (fake=0, invalid=1, real=2)
        if predicted.item() == 0:
            label = 'Counterfeit Money'
        elif predicted.item() == 1:
            label = 'Invalid (Not a Banknote)'
        else:  # predicted.item() == 2
            label = 'Real Money'
            
        confidence_score = confidence.item() * 100  # Convert confidence to percentage
        return {
            'label': label,
            'confidence': confidence_score
        }
    except Exception as e:
        # Handle errors during image processing
        print(f"Error analyzing image: {e}")
        return {
            'label': 'Error',
            'confidence': 0
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle main page and single image upload"""
    if request.method == 'POST':
        # Get uploaded file
        file = request.files['file']
        filename = secure_filename(file.filename)  # Sanitize filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Create file path
        file.save(filepath)  # Save uploaded file

        # Analyze the image
        result = analyze_image(filepath)
        
        # Record timestamp
        timestamp = datetime.datetime.now().strftime('%I:%M %p %m/%d/%Y')

        # Add result to history (insert at beginning)
        history.insert(0, {
            'filename': filename,
            'label': result['label'],
            'confidence': round(result['confidence'], 2),
            'timestamp': timestamp
        })

        # Render result page with analysis details
        return render_template('index.html', filename=filename, label=result['label'], confidence=result['confidence'])

    # Render main page for GET request
    return render_template('index.html')

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Handle batch image uploads and analysis"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400  # Validate file input
    
    files = request.files.getlist('files[]')  # Get list of uploaded files
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400  # Check for valid files
    
    results = []
    
    for file in files:
        filename = secure_filename(file.filename)  # Sanitize filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Create file path
        file.save(filepath)  # Save file
        
        # Analyze the image
        result = analyze_image(filepath)
        timestamp = datetime.datetime.now().strftime('%I:%M %p %m/%d/%Y')  # Record timestamp
        
        # Add to history
        history.insert(0, {
            'filename': filename,
            'label': result['label'],
            'confidence': round(result['confidence'], 2),
            'timestamp': timestamp
        })
        
        # Add to batch results for JSON response
        results.append({
            'filename': filename,
            'label': result['label'],
            'confidence': round(result['confidence'], 2),
            'image_url': url_for('uploaded_file', filename=filename)
        })
    
    # Return JSON response with batch results
    return jsonify({'results': results})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images from the upload folder"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history_page():
    """Render history page with past analysis results"""
    return render_template('history.html', history=history)

@app.route('/about')
def about():
    """Render about page with app information"""
    return render_template('about.html')

@app.route('/download_history')
def download_history():
    """Generate and download history as a CSV file"""
    # Create in-memory CSV buffer
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['filename', 'label', 'confidence', 'timestamp'])
    
    # Write CSV header
    writer.writeheader()
    
    # Write history rows
    for entry in history:
        writer.writerow(entry)
    
    # Prepare CSV for download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'pesocheck_history_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    # Run Flask app in debug mode
    app.run(debug=True)