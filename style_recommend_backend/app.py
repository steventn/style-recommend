import os

from flask import Flask, request, jsonify
from flask_cors import CORS

from src.interface import get_recommendations

app = Flask(__name__)
CORS(app)


@app.route('/profile', methods=['GET'])
def my_profile():
    data = {
        "name": "Steven",
        "message": "Hello! I'm a full stack developer that loves python and javascript Test"
    }

    return jsonify(data)


@app.route('/upload', methods=['POST'])
def upload_file():
    current_directory = os.path.dirname(__file__)

    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # You can save the file to disk or process it further as per your requirements
    print(os.path.join(current_directory, 'uploads', file.filename))
    file.save(os.path.join(current_directory, 'uploads', file.filename))
    return 'File uploaded successfully', 200


@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    image = request.files['file']
    recommendation = get_recommendations(image)
    response = {
        'category': recommendation['predicted_category'],
        'color': recommendation['predicted_color']
    }
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
