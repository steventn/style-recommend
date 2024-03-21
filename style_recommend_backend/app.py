import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.interface import get_recommendations
from src.color_handler import get_color_recommendations

app = Flask(__name__)
CORS(app)


@app.route('/upload', methods=['POST'])
def upload_file():
    current_directory = os.path.dirname(__file__)

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    file.save(os.path.join(current_directory, 'uploads', file.filename))
    return 'File uploaded successfully', 200


@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    image = request.files['file']
    recommendation = get_recommendations(image)
    color_recommendations = get_color_recommendations(recommendation['predicted_color'])
    response = {
        'category': recommendation['predicted_category'],
        'primary_color': recommendation['predicted_color'],
        'color': color_recommendations
    }
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
