from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


class ColorRecommendation:
    def get_color_recommendation(self, image_path):
        # Add your image processing logic here
        # Return the color and complementary color
        color = "example_color"
        complementary_color = "example_complementary_color"
        return color, complementary_color


# @app.route('/get_color_recommendation', methods=['POST'])
# def get_color_recommendation():
#     image_path = request.json.get('image_path')
#
#     recommendation = ColorRecommendation().get_color_recommendation(image_path)
#
#     return jsonify({
#         'color': recommendation[0],
#         'complementary_color': recommendation[1]
#     })


@app.route('/profile')
def my_profile():
    data = {
        "name": "Steven",
        "message" :"Hello! I'm a full stack developer that loves python and javascript Test"
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5050)
