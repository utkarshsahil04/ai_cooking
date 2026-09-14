from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from ultralytics import YOLO
import google.generativeai as genai
import pandas

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Google Gemini API key from environment variable
genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))

# Initialize YOLO model
model = YOLO('yolosaved_bestdataset_2_epoch25.pt')

def detect_items(image_path):
    results = model(image_path)
    detected_items = []
    for result in results:
        for class_id in result.boxes.cls:
            class_name = model.names[int(class_id)]
            detected_items.append(class_name)
    return list(set(detected_items))

def get_recipe_suggestions(detected_items):
    formatted_list = ', '.join(detected_items)
    prompt = f"I have the following fruits and vegetables: {formatted_list}. What food names can I make using them?"

    model_gen = genai.GenerativeModel('gemini-pro')
    response = model_gen.generate_content(prompt)
    return response.text.strip()


@app.route('/')
def index():
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def styles():
    return send_from_directory('', 'styles.css')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    image_path = f'tmp/{image.filename}'
    image.save(image_path)
    
    # Detect items using the YOLO model
    detected_items = detect_items(image_path)
    
    # Get recipe suggestions from Gemini API
    recipe_suggestions = get_recipe_suggestions(detected_items)
    
    return jsonify({
        'detectedItems': detected_items,
        'recipeSuggestions': recipe_suggestions
    })

if __name__ == '__main__':
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    app.run(debug=True)
